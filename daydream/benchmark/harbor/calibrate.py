"""Calibration gate for configured semantic-match judges.

Drives the exact packaged Harbor judge path (``score_review.judge_pairs``,
loaded via importlib from ``templates/tests/`` so the bare ``import
verifier_core`` resolves to the sibling copy) against a fixed 24-pair labeled
fixture, three times per pair (72 judge calls). Computes a three-part pass
gate — majority correctness, balanced accuracy, and per-pair retained-edge
threshold-flip stability — and, on pass, records a private deterministic
invalidation-aware receipt at ``<workspace>/runtime/calibration-receipt.json``.
Measuring, never tuning: no weights or thresholds are derived here.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from daydream.benchmark import storage
from daydream.benchmark.schema import BenchmarkManifest

_HARBOR = Path(__file__).parent
_TEMPLATES = _HARBOR / "templates"
_CALIBRATION_DIR = _HARBOR / "calibration"


def _load_template_asset(path: Path, name: str) -> Any:
    """Load a non-package template asset so a bare sibling import resolves to it."""
    sys.path.insert(0, str(path.parent))  # bare `import verifier_core` -> sibling template copy
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_judge_template() -> Any:
    """Load the packaged ``score_review.py`` judge template as ``score_review``."""
    return _load_template_asset(_TEMPLATES / "tests" / "score_review.py", "score_review")


def _load_fixture() -> list[dict[str, Any]]:
    """Read and return the fixed 24-pair calibration fixture."""
    path = _CALIBRATION_DIR / "pairs.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"could not parse calibration fixture at {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"calibration fixture at {path} must be a JSON list")
    return data


def _build_calibration_client(env: dict[str, Any], *, http: Any = None) -> Any:
    """Build a judge client threading an injectable ``http=`` seam.

    Mirrors ``score_review._build_client``'s provider branch exactly (same
    ``DAYDREAM_JUDGE_*`` reads, same fail-closed provider set, same
    ``_effective_allowlist``/``_validate_base_url``/``resolve_base_url``
    helpers from the loaded module) and additionally passes ``http=http`` into
    the ``AnthropicJudgeClient`` / ``OpenAIJudgeClient`` constructor. With
    ``http=None`` it is behaviorally identical to ``_build_client`` (a real
    httpx client).
    """
    sr = _load_judge_template()
    provider = env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic"
    model = env.get("DAYDREAM_JUDGE_MODEL")
    api_key = env.get("DAYDREAM_JUDGE_API_KEY")
    if not model or not api_key:
        raise sr.VerifierError("missing DAYDREAM_JUDGE_MODEL or DAYDREAM_JUDGE_API_KEY")
    if provider not in {"anthropic", "openai-compatible"}:
        raise sr.VerifierError(
            f"unsupported DAYDREAM_JUDGE_PROVIDER '{provider}'; expected anthropic or openai-compatible"
        )
    if provider == "anthropic":
        allowlist = sr._effective_allowlist(sr._ANTHROPIC_MESSAGES_URL, env)
        sr._validate_base_url(sr._ANTHROPIC_MESSAGES_URL, allowlist)
        return sr.AnthropicJudgeClient(
            api_key, model, http=http, allowlist=allowlist
        )
    base_url = sr.resolve_base_url(api_key, env.get("DAYDREAM_JUDGE_BASE_URL"))
    allowlist = sr._effective_allowlist(base_url, env)
    initial_url = base_url.rstrip("/") + "/chat/completions"
    sr._validate_base_url(initial_url, allowlist)
    base_url = sr.resolve_base_url(api_key, env.get("DAYDREAM_JUDGE_BASE_URL"))
    allowlist = sr._effective_allowlist(base_url, env)
    initial_url = base_url.rstrip("/") + "/chat/completions"
    sr._validate_base_url(initial_url, allowlist)
    return sr.OpenAIJudgeClient(
        api_key, model, base_url=base_url, http=http, allowlist=allowlist
    )


def _judge_host_from_env(env: dict[str, Any]) -> str:
    """Return the normalized-lowercase judge host for ``env``.

    The anthropic provider (or absent -> anthropic default) always routes to
    ``api.anthropic.com``; the openai-compatible provider resolves its base URL
    via the packaged ``resolve_base_url`` and returns that URL's host.
    """
    sr = _load_judge_template()
    provider = env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic"
    if provider == "anthropic":
        return "api.anthropic.com"
    base_url = sr.resolve_base_url(
        env.get("DAYDREAM_JUDGE_API_KEY") or "", env.get("DAYDREAM_JUDGE_BASE_URL")
    )
    return str(urllib.parse.urlsplit(base_url).hostname or "").lower()


def _load_workspace_allowlist(workspace: Path) -> list[str]:
    """Read ``<workspace>/benchmark.yaml`` and return its judge allowlist.

    Validates the manifest strictly via ``BenchmarkManifest.model_validate``;
    a malformed or missing manifest raises the project's existing workspace
    error (``WorkspaceCorrupt``) — never a silent default.
    """
    try:
        raw = storage.load_yaml_strict(workspace / "benchmark.yaml")
    except storage.WorkspaceCorrupt:
        raise
    try:
        manifest = BenchmarkManifest.model_validate(raw)
    except Exception as exc:
        raise storage.WorkspaceCorrupt(
            f"{workspace}: invalid benchmark.yaml: {exc}"
        ) from exc
    return list(manifest.privacy.judge_allowed_hosts)


def _validate_workspace_host(allowlist: list[str] | set[str], host: str) -> None:
    """Fail closed when ``host`` is not in the workspace judge host allowlist."""
    if host not in allowlist:
        raise ValueError(
            f"judge host {host!r} is not in the workspace judge_allowed_hosts allowlist"
        )


def _judge_pairs(
    template: Any, client: Any, pairs: list[dict[str, Any]], *, attempts: int = 3
) -> list[list[Any]]:
    """Drive the packaged ``judge_pairs`` three times per pair, in fixture order.

    Each invocation is 1 gold x 1 candidate, so it makes exactly one judge call;
    24 pairs x ``attempts`` calls for the exact 72-call budget. Judge-call
    failures propagate via the packaged module's ``VerifierError`` — never a
    silent fallback (a judge-call failure aborts the run, fail-closed).
    """
    per_pair: list[list[Any]] = []
    for pair in pairs:
        runs: list[Any] = []
        for _ in range(attempts):
            verdicts = asyncio.run(
                template.judge_pairs(
                    gold=[pair["gold"]],
                    candidates=[pair["candidate"]],
                    client=client,
                )
            )
            runs.append(verdicts[0])
        per_pair.append(runs)
    return per_pair


def _majority_label(verdicts: list[Any]) -> bool:
    """Return the majority ``match`` over a pair's runs; ties resolve to False."""
    matches = sum(1 for v in verdicts if v.match)
    return matches * 2 > len(verdicts)


def _retained_edge(v: Any, threshold: float) -> bool:
    """Return True when a verdict is a retained edge: match and confident."""
    return bool(v.match and v.confidence >= threshold)


def _per_pair_stable(verdicts: list[Any], threshold: float) -> bool:
    """True when the retained-edge boolean sequence flips at most once."""
    retained = [_retained_edge(v, threshold) for v in verdicts]
    flips = sum(1 for a, b in zip(retained, retained[1:]) if a != b)
    return flips <= 1


def _confusion_matrix(
    gold_labels: list[Any], majority_labels: list[bool]
) -> dict[str, int]:
    """Standard 2x2 confusion counts for a match/nonmatch label set.

    Gold labels may be booleans (``True`` = match) or the strings
    ``"match"`` / ``"nonmatch"``.
    """
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for truth, pred in zip(gold_labels, majority_labels):
        actual = truth is True or truth == "match"
        if actual and pred:
            counts["tp"] += 1
        elif not actual and pred:
            counts["fp"] += 1
        elif not actual and not pred:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def _balanced_accuracy(
    matches_correct: int,
    matches_wrong: int,
    nonmatches_correct: int,
    nonmatches_wrong: int,
) -> float:
    """Balanced accuracy over the periodic 12-match / 12-nonmatch split."""
    return 0.5 * (matches_correct / 12 + nonmatches_correct / 12)


def _class_balanced_accuracy(matrix: dict[str, int]) -> float:
    """Recognition average over the two classes actually present.

    For the full 24-pair fixture (12 match / 12 nonmatch) this equals
    ``0.5 * (tp/12 + tn/12)``; per-class denominators keep small partial
    gates (e.g. a 3-pair unit test) from spuriously failing the fixture's
    hard-coded 12-per-class denominator.
    """
    match_total = matrix["tp"] + matrix["fn"]
    nonmatch_total = matrix["tn"] + matrix["fp"]
    match_recall = matrix["tp"] / match_total if match_total else 1.0
    nonmatch_recall = matrix["tn"] / nonmatch_total if nonmatch_total else 1.0
    return 0.5 * (match_recall + nonmatch_recall)


def _pass_gate(
    pairs: list[dict[str, Any]],
    runs_per_pair: list[list[Any]],
    threshold: float,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate the three-part pass gate, returning (passed, failures).

    ``failures`` is keyed by condition name and empty on pass; never writes
    anything — a pure computation over the already-validated verdicts.
    """
    failures: dict[str, Any] = {}
    gold_labels = [p["label"] for p in pairs]
    majority_labels = [_majority_label(runs) for runs in runs_per_pair]

    wrong_pairs = [
        i
        for i, (truth, pred) in enumerate(zip(gold_labels, majority_labels))
        if (truth == "match") != bool(pred)
    ]
    if wrong_pairs:
        failures["majority"] = wrong_pairs

    matrix = _confusion_matrix(gold_labels, majority_labels)
    bacc = _class_balanced_accuracy(matrix)
    if bacc < 0.9:
        failures["balanced_accuracy"] = bacc

    unstable = [
        i for i, runs in enumerate(runs_per_pair) if not _per_pair_stable(runs, threshold)
    ]
    if unstable:
        failures["instability"] = unstable

    return (len(failures) == 0, failures)


def _label_sha256(pairs: list[dict[str, Any]]) -> str:
    """Canonical sha256 over the ordered gold/candidate/label triples."""
    triples = [
        {
            "gold": p["gold"]["finding_id"],
            "candidate": p["candidate"]["candidate_id"],
            "label": p["label"],
        }
        for p in pairs
    ]
    canonical = json.dumps(
        triples, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _render_judge_prompt_digest(sr: Any) -> str:
    """sha256 of the packaged judge prompt template bytes."""
    prompt_path = _TEMPLATES / "tests" / "judge_prompt.md"
    return hashlib.sha256(prompt_path.read_bytes()).hexdigest()


def _serialize_inputs(inputs: dict[str, Any]) -> bytes:
    """Deterministic byte-stable serialization of the invalidation inputs."""
    return json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _invalidation_inputs(
    env: dict[str, Any], pairs: list[dict[str, Any]], sr: Any
) -> dict[str, Any]:
    """The receipt's invalidation contract: a deterministic byte-stable dict."""
    return {
        "provider": env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic",
        "model": env.get("DAYDREAM_JUDGE_MODEL") or "",
        "host": _judge_host_from_env(env),
        "judge_prompt_sha256": _render_judge_prompt_digest(sr),
        "threshold": sr.verifier_core.CONFIDENCE_THRESHOLD,
        "label_sha256": _label_sha256(pairs),
        "attempts": 3,
        "request_timeout": sr._REQUEST_TIMEOUT,
    }


def _build_receipt(
    sr: Any,
    pairs: list[dict[str, Any]],
    env: dict[str, Any],
    *,
    attempts: int,
    passed: bool,
    balanced_accuracy: float,
    confusion: dict[str, int],
    disagreements: list[Any],
) -> dict[str, Any]:
    """Build the private deterministic calibration receipt document."""
    return {
        "schema_version": 1,
        "type": "judge-calibration",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "inputs": _invalidation_inputs(env, pairs, sr),
        "result": {
            "passed": passed,
            "balanced_accuracy": balanced_accuracy,
            "confusion_matrix": confusion,
            "disagreements": sorted(disagreements, key=lambda d: d["pair_index"]),
        },
    }


def _write_receipt(workspace: Path, receipt: dict[str, Any]) -> Path:
    """Atomically write the calibration receipt private (0o600) and return its path."""
    runtime = Path(workspace) / "runtime"
    path = runtime / "calibration-receipt.json"
    storage.atomic_write_json(path, receipt, mode=0o600)
    return path


def is_receipt_current(receipt_path: Path, current_inputs: dict[str, Any]) -> bool:
    """Fail-closed invalidation check: True only on a byte-exact inputs match."""
    try:
        raw = json.loads(Path(receipt_path).read_bytes())
    except (ValueError, OSError):
        return False
    if not isinstance(raw, dict) or not isinstance(raw.get("inputs"), dict):
        return False
    return _serialize_inputs(raw["inputs"]) == _serialize_inputs(current_inputs)


def _default_confirm(prompt: str) -> bool:
    """Read a confirmation reply from stdin; non-interactive stdin is refused."""
    reply = input(prompt)
    return reply.strip().lower() in ("y", "yes", "")


def run_calibration(
    workspace: Path,
    *,
    yes: bool = False,
    env: dict[str, Any] | None = None,
    http: Any = None,
    confirm: Callable[[str], bool] | None = None,
) -> int:
    """Drive the calibration gate end-to-end and return an exit code.

    Order: manifest/host validation -> fixture + template load -> confirmation
    gate -> client build -> 72-call judge driver -> three-part pass gate.
    On pass a private receipt is written; on failure (or refusal) no receipt is
    written and the exit code is nonzero (fail-closed). Never measures by
    tuning anything.
    """
    import sys as _sys

    from daydream.benchmark.storage import WorkspaceCorrupt as _WorkspaceCorrupt

    env = dict(env) if env is not None else dict(os.environ)
    sr = _load_judge_template()
    try:
        allowlist = _load_workspace_allowlist(workspace)
    except _WorkspaceCorrupt as exc:
        print(str(exc), file=_sys.stderr)
        return 1
    host = _judge_host_from_env(env)
    try:
        _validate_workspace_host(allowlist, host)
    except ValueError as exc:
        print(str(exc), file=_sys.stderr)
        return 1

    pairs = _load_fixture()
    inputs = _invalidation_inputs(env, pairs, sr)

    if not yes:
        if confirm is None:
            confirm = _default_confirm
        prompt = (
            f"Run calibration against {inputs['provider']} model "
            f"{inputs['model']!r} on host {inputs['host']}? "
            f"judge prompt digest {inputs['judge_prompt_sha256'][:12]}, "
            f"threshold {inputs['threshold']}, 72 judged calls, "
            f"request timeout {inputs['request_timeout']}s; this incurs paid "
            f"API costs. Proceed? [y/N] "
        )
        if not confirm(prompt):
            print("calibrate-judge: aborted before any paid call", file=_sys.stderr)
            return 1

    try:
        client = _build_calibration_client(env, http=http)
        runs = _judge_pairs(sr, client, pairs, attempts=3)
        passed, failures = _pass_gate(
            pairs, runs, sr.verifier_core.CONFIDENCE_THRESHOLD
        )
    except sr.VerifierError as exc:
        print(f"calibrate-judge: judge failure: {sr._bounded_error(str(exc))}", file=_sys.stderr)
        return 1

    if passed:
        matrix = _confusion_matrix(
            [p["label"] for p in pairs], [_majority_label(runs) for runs in runs]
        )
        bacc = _class_balanced_accuracy(matrix)
        disagreements = [
            {
                "pair_index": i,
                "gold_id": p["gold"]["finding_id"],
                "candidate_id": p["candidate"]["candidate_id"],
                "per_run_retained": [
                    _retained_edge(v, sr.verifier_core.CONFIDENCE_THRESHOLD) for v in r
                ],
            }
            for i, (p, r) in enumerate(zip(pairs, runs))
        ]
        receipt = _build_receipt(
            sr, pairs, env, attempts=3, passed=True,
            balanced_accuracy=bacc, confusion=matrix, disagreements=disagreements,
        )
        _write_receipt(workspace, receipt)
        print(
            f"calibrate-judge: PASS (balanced accuracy {bacc:.4f}); "
            f"receipt written to {workspace / 'runtime' / 'calibration-receipt.json'}"
        )
        return 0

    for condition, detail in failures.items():
        print(f"calibrate-judge: FAIL ({condition}): {detail}", file=_sys.stderr)
    return 1

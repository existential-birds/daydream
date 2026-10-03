"""Read-only, attributable objectives for exact completed Harbor runs. Reuse
ledger/containment validation, strictly parse task rewards, and pool count-derived
metrics. Frozen results never modify run state.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeGuard

from daydream.benchmark.harbor import calibrate, run as run_mod, verifier_core


class ObjectiveError(Exception):
    """A missing/incomplete run or malformed ledger, identity, or reward artifact, labeled
    by run and path.
    """


@dataclass(frozen=True)
class Objective:
    """Canonical pooled metrics plus completion accounting for one native run.

    Any unscored trial makes the result ineligible for comparison. Keep the
    scorer's metric names and types until the existing per-run wire projection.
    """

    metrics: Mapping[str, float | int]
    clean_pass_count: int
    candidate_count: int
    gold_count: int
    verifier_error_task_count: int
    malformed_task_count: int
    failed_task_count: int
    comparison_eligible: bool
    tokens: float | None = None
    cost: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class CompatibilityIdentity:
    """Compatibility fields bound from recorded ledger, lock, runtime, and scorer sources.
    Missing reviewer effort remains None; attribution is never inferred.
    """

    objective_schema_version: int
    profile_schema_version: int
    profile_name: str
    profile_digest: str | None
    daydream_version: str
    daydream_wheel_sha256: str
    compiled_lock_sha256: str
    harbor_version: str
    reviewer_backend: str
    reviewer_model: str
    reviewer_base_url: str
    reviewer_effort: str | None
    judge_provider: str
    judge_model: str
    judge_host: str
    verifier_template_sha256: str
    threshold: float
    attempts: int


@dataclass(frozen=True)
class SuiteEntry:
    """One exact completion referenced by a suite manifest."""

    workspace: Path
    run_id: str


def identity_to_dict(identity: CompatibilityIdentity) -> dict[str, object]:
    """Project the identity dataclass consistently; repository and benchmark ids are
    intentionally excluded.
    """
    return asdict(identity)


@dataclass(frozen=True)
class SuiteObjective:
    """Pooled metrics for a fully compatible suite of exact completions. Experiment
    identity hashes the canonical manifest and shared compatibility identity. Invalid
    entries raise before a result exists; diagnostics is empty.
    """

    objective: Objective
    experiment_id: str
    profile_digest: str | None
    identity: CompatibilityIdentity
    diagnostics: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class CompletedRun:
    """Immutable projection of a completed, ledgered benchmark run."""

    run_id: str
    mode: str
    state: str
    # Full compatibility identity from authoritative sources.
    identity: CompatibilityIdentity | None = None
    # Per-task reward rows (flattened).
    task_rows: list[dict[str, object] | None] = field(default_factory=list)
    # Count-derived objective.
    objective: Objective | None = None


def read_completed_run(
    workspace: Path, run_id: str, *, env: dict[str, Any] | None = None
) -> CompletedRun:
    """Resolve one explicit completed run with authoritative identity and strict task
    rewards. Missing/noncomplete runs, malformed artifacts, and identity disagreements
    raise ObjectiveError naming the run and offending artifact.
    """
    env = env or {}
    try:
        doc = run_mod._load_ledger(workspace)
    except run_mod.RunError as exc:
        raise ObjectiveError(f"ledger failure at {workspace}: {exc}") from exc

    entry = None
    for run in doc["runs"]:
        if run.get("run_id") == run_id:
            entry = run
            break

    if entry is None:
        raise ObjectiveError(
            f"run {run_id!r} not found in the harbor ledger at {workspace}"
        )
    if entry.get("state") != "complete":
        raise ObjectiveError(
            f"run {run_id!r} at {workspace} is not complete "
            f"(state {entry.get('state')!r})"
        )

    try:
        job_dir = run_mod._validate_job_dir(workspace, str(entry.get("job_dir") or ""))
    except run_mod.RunError as exc:
        raise ObjectiveError(
            f"run {run_id!r} at {workspace} has uncontained job_dir: {exc}"
        ) from exc
    identity = _bind_identity(workspace, entry, run_id, env)
    rows, infra_errors = _parse_task_rows(Path(job_dir), run_id)
    objective = _build_objective(rows, infra_errors, Path(job_dir), run_id)

    return CompletedRun(
        run_id=entry["run_id"],
        mode=entry["mode"],
        state=entry["state"],
        identity=identity,
        task_rows=rows,
        objective=objective,
    )


# The schema version recorded in ``CompatibilityIdentity.objective_schema_version``.
_OBJECTIVE_SCHEMA_VERSION = 1


def objective_to_json(run: CompletedRun) -> dict[str, object]:
    """Emit only opaque run identity and metrics, dropping filesystem and evidence content.
    No repository slug, PR number, source path, text, reasoning, or source code is
    included; the ledger job_dir is deliberately omitted.
    """
    identity = run.identity
    identity_json = None
    if identity is not None:
        identity_json = identity_to_dict(identity)

    objective_dict: dict[str, object] | None = None
    if run.objective is not None:
        obj = run.objective
        objective_dict = {
            item.name: getattr(obj, item.name)
            for item in fields(Objective)
            if item.name != "metrics"
        }
        metrics = dict(obj.metrics)
        for wire_name, metric_name in (
            ("tp", "total_tp"),
            ("fp", "total_fp"),
            ("fn", "total_fn"),
            ("precision", "micro_precision"),
            ("recall", "micro_recall"),
            ("f1", "micro_f1"),
        ):
            metrics[wire_name] = metrics.pop(metric_name)
        objective_dict.update(metrics)
        if obj.tokens is None:
            del objective_dict["tokens"]
        if obj.cost is None:
            del objective_dict["cost"]

    return {
        "run_id": run.run_id,
        "mode": run.mode,
        "schema_version": _OBJECTIVE_SCHEMA_VERSION,
        "identity": identity_json,
        "objective": objective_dict,
    }


def _canonical_suite_manifest(entries: list[SuiteEntry]) -> dict[str, object]:
    """Canonical, reorder-stable projection of a validated suite manifest."""
    return {
        "schema_version": _SUITE_SCHEMA_VERSION,
        "entries": sorted(
            ({"workspace": str(e.workspace), "run_id": e.run_id} for e in entries),
            key=lambda ent: (ent["workspace"], ent["run_id"]),
        ),
    }


def _suite_experiment_id(
    entries: list[SuiteEntry], identity: CompatibilityIdentity
) -> str:
    """Stable SHA-256 over the canonicalized manifest plus the shared identity."""
    payload = {
        "manifest": _canonical_suite_manifest(entries),
        "identity": identity_to_dict(identity),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def aggregate_suite(
    manifest: dict[str, Any], *, env: dict[str, Any] | None = None
) -> SuiteObjective:
    """Validate every exact completion and require identical compatibility across the
    suite. Reject incomplete, malformed, duplicated, or comparison-ineligible entries;
    never return a subset. Pool flattened task rows once through aggregate_metrics, so
    precision/recall/F1 derive from counts rather than repository averages.
    """
    env = env or {}
    entries = validate_suite_manifest(manifest)

    resolved: list[tuple[SuiteEntry, CompletedRun]] = []
    for index, entry in enumerate(entries):
        try:
            run = read_completed_run(entry.workspace, entry.run_id, env=env)
        except ObjectiveError as exc:
            raise ObjectiveError(f"suite entry #{index} failed: {exc}") from exc
        resolved.append((entry, run))

    identities: list[CompatibilityIdentity | None] = [r.identity for _, r in resolved]
    if any(identity is None for identity in identities):
        raise ObjectiveError(
            "suite entries must each bind a compatibility identity for pooling"
        )
    base = identities[0]
    assert base is not None
    base_fields = identity_to_dict(base)
    for entry, run in resolved[1:]:
        run_identity = run.identity
        assert run_identity is not None
        for comp_field, value in base_fields.items():
            if getattr(run_identity, comp_field) != value:
                raise ObjectiveError(
                    f"suite is not comparable: {comp_field} differs across entries "
                    f"at workspace {entry.workspace} run {entry.run_id!r}"
                )

    # Fail closed on any comparison-ineligible entry (never a subsetted pool).
    for entry, run in resolved:
        if run.objective is not None and not run.objective.comparison_eligible:
            raise ObjectiveError(
                f"suite entry at workspace {entry.workspace} run {entry.run_id!r} "
                f"is not comparison-eligible; refusing to pool"
            )

    rows = [row for _, run in resolved for row in run.task_rows]
    infra_errors = sum(1 for row in rows if row is None)
    suite_label = _suite_label(entries)
    pooled = _build_objective(
        rows, infra_errors, job_dir=Path("@suite"), run_id=suite_label
    )
    experiment_id = _suite_experiment_id(entries, base)
    return SuiteObjective(
        objective=pooled,
        experiment_id=experiment_id,
        profile_digest=base.profile_digest,
        identity=base,
    )


def _suite_label(entries: list[SuiteEntry]) -> str:
    return "-".join(f"{e.workspace.name}:{e.run_id}" for e in entries)


_PROFILE_SCHEMA_VERSION = 1

_SUITE_SCHEMA_VERSION = 1


def validate_suite_manifest(manifest: dict[str, Any]) -> list[SuiteEntry]:
    """Require schema 1 and well-formed, unique workspace/run pairs; preserve order and
    identify bad entries.
    """
    if not isinstance(manifest, dict):
        raise ObjectiveError("suite manifest must be a mapping object")
    if manifest.get("schema_version") != _SUITE_SCHEMA_VERSION:
        raise ObjectiveError(
            f"suite manifest has unsupported schema_version "
            f"{manifest.get('schema_version')!r}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ObjectiveError("suite manifest is missing its entries list")

    result: list[SuiteEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ObjectiveError(
                f"suite manifest entry #{index} is not an object: {raw!r}"
            )
        missing = [key for key in ("workspace", "run_id") if not raw.get(key)]
        if missing:
            raise ObjectiveError(
                f"suite manifest entry #{index} is missing field(s) "
                f"{', '.join(repr(k) for k in missing)}"
            )
        workspace = Path(str(raw["workspace"]))
        run_id = str(raw["run_id"])
        pair = (str(workspace), run_id)
        if pair in seen:
            raise ObjectiveError(
                f"suite manifest entry #{index} duplicates (workspace={workspace!s}, "
                f"run_id={run_id!r})"
            )
        seen.add(pair)
        result.append(SuiteEntry(workspace=workspace, run_id=run_id))
    return result


def _bind_identity(
    workspace: Path, entry: dict[str, Any], run_id: str, env: dict[str, Any]
) -> CompatibilityIdentity:
    """Bind identity only from authoritative artifacts, requiring the ledger exact lock
    digest. Unreadable, missing, malformed, or mismatched provenance raises
    ObjectiveError.
    """
    ledger_digest = entry.get("compiled_lock_sha256")
    try:
        disk_digest = run_mod._compiled_lock_sha256(workspace)
    except OSError as exc:
        raise ObjectiveError(
            f"run {run_id!r}: cannot hash compiled lock at "
            f"{workspace / 'harbor' / 'benchmark.lock.json'}: {exc}"
        ) from exc
    if ledger_digest != disk_digest:
        raise ObjectiveError(
            f"run {run_id!r} ledger compiled_lock_sha256 disagrees with the "
            f"on-disk compiled lock at {workspace / 'harbor' / 'benchmark.lock.json'}"
        )

    try:
        wheel_version, wheel_sha = run_mod._compiled_daydream_wheel(workspace)
    except run_mod.RunError as exc:
        raise ObjectiveError(f"run {run_id!r}: {exc}") from exc

    try:
        harbor_version = ".".join(
            str(importlib.metadata.version("harbor")).split(".")[:2]
        )
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise ObjectiveError(f"run {run_id!r}: harbor package metadata not found") from exc

    judge_template = calibrate._load_judge_template()
    profile_digest = entry.get("profile_digest") or env.get(
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"
    )
    try:
        attempts = int(run_mod._compiled_job_config(workspace).get("n_attempts", 1))
    except (run_mod.RunError, ValueError, TypeError) as exc:
        raise ObjectiveError(
            f"run {run_id!r}: cannot read compiled harbor-job.yaml at "
            f"{workspace / 'harbor' / 'harbor-job.yaml'}: {exc}"
        ) from exc

    return CompatibilityIdentity(
        objective_schema_version=_OBJECTIVE_SCHEMA_VERSION,
        profile_schema_version=_PROFILE_SCHEMA_VERSION,
        profile_name="",
        profile_digest=str(profile_digest) if profile_digest else None,
        daydream_version=str(wheel_version),
        daydream_wheel_sha256=str(wheel_sha),
        compiled_lock_sha256=str(ledger_digest),
        harbor_version=harbor_version,
        reviewer_backend=entry.get("reviewer_backend")
        or env.get("DAYDREAM_REVIEW_BACKEND")
        or "",
        reviewer_model=entry.get("reviewer_model")
        or env.get("DAYDREAM_REVIEW_MODEL")
        or "",
        reviewer_base_url=entry.get("reviewer_base_url")
        or env.get("DAYDREAM_REVIEW_BASE_URL")
        or "",
        # Recorded at run-append time; absent -> None (never fabricated).
        reviewer_effort=entry.get("reviewer_effort"),
        judge_provider=entry.get("judge_provider")
        or env.get("DAYDREAM_JUDGE_PROVIDER")
        or "",
        judge_model=entry.get("judge_model") or env.get("DAYDREAM_JUDGE_MODEL") or "",
        judge_host=entry.get("judge_host")
        or (
            calibrate._judge_host_from_env(env)
            if env.get("DAYDREAM_JUDGE_PROVIDER")
            else ""
        ),
        verifier_template_sha256=calibrate._render_judge_prompt_digest(judge_template),
        threshold=verifier_core.CONFIDENCE_THRESHOLD,
        attempts=attempts,
    )


# The integer count keys a scored task must carry as JSON integers (mirrors
# the generated reward-row shape consumed by ``verifier_core.aggregate_metrics``).
_SCORED_COUNT_KEYS = ("tp", "fp", "fn")


def _parse_task_rows(
    job_dir: Path, run_id: str
) -> tuple[list[dict[str, object] | None], int]:
    """Read sorted trials: strict reward.json means scored; details-only means an infra
    failure. Unscored rows become None, never numeric zero. A valid empty clean task
    remains scored. Malformed reward artifacts fail the whole read.
    """
    if not job_dir.is_dir():
        return [], 0
    rows: list[dict[str, object] | None] = []
    infra_errors = 0
    for trial in run_mod._iter_trial_dirs(job_dir):
        verifier = trial / "verifier"
        reward_path = verifier / "reward.json"
        if reward_path.is_file():
            row = _parse_reward_strict(reward_path, run_id)
            rows.append(row)
        elif (verifier / "reward-details.json").is_file():
            # Unscored infra trial (never a numeric zero).
            rows.append(None)
            infra_errors += 1
        else:
            raise ObjectiveError(
                f"trial {trial.name} in run {run_id!r} has no score evidence at "
                f"{reward_path}"
            )
    return rows, infra_errors


def _parse_reward_strict(reward_path: Path, run_id: str) -> dict[str, object]:
    """Reject non-object rewards, invalid/nonfinite/negative counts, and
    nonnumeric/nonfinite reward values.
    """
    try:
        data: dict[str, object] = json.loads(reward_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectiveError(
            f"malformed reward artifact at {reward_path} in run {run_id!r}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ObjectiveError(
            f"reward artifact at {reward_path} in run {run_id!r} is not an object"
        )

    for key in _SCORED_COUNT_KEYS:
        value = data.get(key)
        if not _is_int(value):
            raise ObjectiveError(
                f"reward artifact at {reward_path} in run {run_id!r} has "
                f"non-integer {key!r}: {value!r}"
            )
        if value < 0:
            raise ObjectiveError(
                f"reward artifact at {reward_path} in run {run_id!r} has "
                f"negative {key!r}: {value!r}"
            )

    reward = data.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise ObjectiveError(
            f"reward artifact at {reward_path} in run {run_id!r} has "
            f"non-numeric reward: {reward!r}"
        )
    if not _is_finite(reward):
        raise ObjectiveError(
            f"reward artifact at {reward_path} in run {run_id!r} has "
            f"non-finite reward: {reward!r}"
        )
    return data


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _row_int(row: dict[str, object], key: str) -> int:
    """Read an integer-valued row field, defaulting to 0 when absent."""
    value = row.get(key)
    return value if _is_int(value) else 0


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _build_objective(
    rows: list[dict[str, object] | None], infra_errors: int,
    job_dir: Path, run_id: str,
) -> Objective:
    """Pool canonical scorer metrics and verifier-error counts; any unscored trial blocks
    comparison.
    """
    try:
        agg = verifier_core.aggregate_metrics(rows)
    except verifier_core.VerifierError as exc:
        raise ObjectiveError(
            f"malformed reward row(s) in run {run_id!r} under {job_dir}: {exc}"
        ) from exc
    verifier_errors = len([
        row for row in rows
        if row is not None and _row_int(row, "verifier_error") == 1
    ])
    # Malformed rows fail closed in ``_parse_reward_strict`` and so never reach
    # this pool; the count stays zero by construction.
    malformed = 0
    failed = max(verifier_errors, infra_errors)
    candidate_count = sum(
        _row_int(row, "candidate_count") for row in rows if row is not None
    )
    gold_count = sum(
        _row_int(row, "gold_count") for row in rows if row is not None
    )
    clean_task_count = int(agg["clean_task_count"])
    return Objective(
        metrics=agg,
        clean_pass_count=int(round(agg["clean_accuracy"] * clean_task_count)),
        candidate_count=candidate_count,
        gold_count=gold_count,
        verifier_error_task_count=verifier_errors,
        malformed_task_count=malformed,
        failed_task_count=failed,
        comparison_eligible=bool(agg["scored_task_count"]) and not (
            infra_errors + verifier_errors + malformed + failed
        ),
    )

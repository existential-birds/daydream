"""Read-only resolution of exact completed Harbor benchmark runs (issue #888).

A future optimizer can consume an attributable, machine-readable Harbor
objective for any exact completed run. This module is the read-only surface:
it resolves a ledgered run by explicit ``run_id``, validates it reached a
terminal ``complete`` state, parses its per-task reward artifacts in strict
fail-closed fashion, and binds its count-derived micro-metrics.

Everything here is strictly read-only and immutable. State is modelled with
frozen dataclasses; ``_load_ledger`` and ``_validate_job_dir`` are reused from
``run`` so the reader never re-implements ledger parsing, validation, or job
dir containment — it only filters on ``state`` and reads the reward rows.
"""
from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import calibrate, verifier_core
from daydream.benchmark.harbor import run as run_mod


class ObjectiveError(Exception):
    """The single typed error for objective resolution.

    Raised when a run is absent from the ledger, is not in a terminal
    ``complete`` state, the ledger fails to parse/validate, or any per-task
    reward artifact is malformed. The message always names the offending
    artifact path and run id.
    """


@dataclass(frozen=True)
class Objective:
    """Count-derived micro-metrics for a single completed run.

    TP/FP/FN are pooled (never per-task averaged) along with the clean/task/
    infra counts by feeding the flattened per-task reward rows through
    ``verifier_core.aggregate_metrics`` — the same authoritative scorer the
    generated corpus uses. ``comparison_eligible`` is ``False`` whenever any
    trial failed to produce a legitimate scored row and must not participate
    in comparison.
    """

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    clean_task_count: int
    clean_pass_count: int
    clean_accuracy: float
    task_count: int
    scored_task_count: int
    candidate_count: int
    gold_count: int
    infra_error_task_count: int
    verifier_error_task_count: int
    malformed_task_count: int
    failed_task_count: int
    comparison_eligible: bool
    mean_task_score: float
    tokens: float | None = None
    cost: float | None = None

    def _as_metric_dict(self) -> dict[str, float | int]:
        """Project this objective in the exact shape ``aggregate_metrics`` returns.

        The count-derived fields were bound by feeding the run's flattened rows
        through ``verifier_core.aggregate_metrics`` exactly once, so this maps
        them back into the authoritative scorer's key/shaper set for byte-for-
        value comparison against the generated ``metric.py`` aggregation.
        """
        return {
            "micro_precision": self.precision,
            "micro_recall": self.recall,
            "micro_f1": self.f1,
            "mean_task_score": self.mean_task_score,
            "clean_accuracy": self.clean_accuracy,
            "task_count": self.task_count,
            "scored_task_count": self.scored_task_count,
            "infra_error_task_count": self.infra_error_task_count,
            "clean_task_count": self.clean_task_count,
            "total_tp": self.tp,
            "total_fp": self.fp,
            "total_fn": self.fn,
        }


@dataclass(frozen=True)
class CompatibilityIdentity:
    """Frozen, attributable compatibility identity of one exact completed run.

    Every field is bound from an authoritative source only (the ledger entry,
    the compiled ``benchmark.lock.json`` ``daydream`` block, the trusted
    control-plane env, ``verifier_core``/``calibrate`` constants, and the
    compiled ``harbor-job.yaml`` attempts) — never from inference or coercion.
    ``reviewer_effort`` is the ledger-recorder effort (read-only) or ``None``
    when absent; it is never fabricated.
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
class CompletedRun:
    """Immutable projection of a completed, ledgered benchmark run."""

    run_id: str
    mode: str
    state: str
    # Full compatibility identity from authoritative sources.
    identity: CompatibilityIdentity | None = None
    # Per-task reward rows (flattened); populated in Task 2.
    task_rows: list[dict[str, object] | None] = field(default_factory=list)
    # Count-derived objective (populated in Task 2).
    objective: Objective | None = None
    # The validated, contained job dir this run executed under.
    job_dir: str = ""


def read_completed_run(
    workspace: Path, run_id: str, *, env: dict[str, Any] | None = None
) -> CompletedRun:
    """Resolve a ledgered run by explicit ``run_id``.

    Loads the ledger through ``run_mod._load_ledger`` and admits only a run
    whose ``state == "complete"``, binds its full compatibility identity from
    authoritative sources, then parses its per-task reward rows in strict
    fail-closed fashion and binds the count-derived ``Objective``.
    Missing runs, running/cleanup-pending/cleaned runs, any ledger parse/
    validation failure, any identity disagreement, and any malformed reward
    artifact all fail closed with an ``ObjectiveError`` naming the run id and
    the offending artifact.
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

    job_dir = run_mod._validate_job_dir(workspace, str(entry.get("job_dir") or ""))
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
        job_dir=job_dir,
    )


# The schema version recorded in ``CompatibilityIdentity.objective_schema_version``.
_OBJECTIVE_SCHEMA_VERSION = 1


# The review-profile schema version this plan's identity binds (the current
# ``ReviewProfile.schema_version``). There is no read-only source for a loaded
# profile's ``name`` here, so it stays empty rather than fabricated.
_PROFILE_SCHEMA_VERSION = 1


def _bind_identity(
    workspace: Path, entry: dict[str, Any], run_id: str, env: dict[str, Any]
) -> CompatibilityIdentity:
    """Bind the full compatibility identity from authoritative sources.

    The ledger's ``compiled_lock_sha256`` must equal the hash of the on-disk
    compiled lock (oracle/default-run gate contract); disagreement is corruption
    and raises ``ObjectiveError`` naming the offending field. Every other
    fallible read (lock parse, missing/malformed ``daydream`` wheel block,
    compiled job config) propagates via ``ObjectiveError`` naming the artifact
    path — no plausible placeholder is ever defaulted.
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

    lock = _load_compiled_lock(workspace, run_id)
    day = lock.get("daydream")
    if not isinstance(day, dict):
        raise ObjectiveError(
            f"run {run_id!r}: compiled lock at {workspace / 'harbor' / 'benchmark.lock.json'}"
            f" is missing its 'daydream' wheel block"
        )
    wheel_sha = day.get("sha256")
    wheel_version = day.get("version")
    if not isinstance(wheel_sha, str) or not wheel_sha or not isinstance(wheel_version, str):
        raise ObjectiveError(
            f"run {run_id!r}: compiled lock at {workspace / 'harbor' / 'benchmark.lock.json'}"
            f" has a malformed or missing 'daydream' wheel digest/version"
        )

    try:
        harbor_version = ".".join(
            str(importlib.metadata.version("harbor")).split(".")[:2]
        )
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise ObjectiveError(f"run {run_id!r}: harbor package metadata not found") from exc

    judge_template = calibrate._load_judge_template()
    profile_digest = env.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST") or entry.get(
        "profile_digest"
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
        reviewer_backend=env.get("DAYDREAM_REVIEW_BACKEND") or "",
        reviewer_model=env.get("DAYDREAM_REVIEW_MODEL") or "",
        reviewer_base_url=env.get("DAYDREAM_REVIEW_BASE_URL") or "",
        # Recorded at run-append time; absent -> None (never fabricated).
        reviewer_effort=entry.get("reviewer_effort"),
        judge_provider=env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic",
        judge_model=env.get("DAYDREAM_JUDGE_MODEL") or "",
        judge_host=calibrate._judge_host_from_env(env),
        verifier_template_sha256=calibrate._render_judge_prompt_digest(judge_template),
        threshold=verifier_core.CONFIDENCE_THRESHOLD,
        attempts=attempts,
    )


def _load_compiled_lock(workspace: Path, run_id: str) -> dict[str, Any]:
    """Read the compiled ``harbor/benchmark.lock.json`` strictly."""
    path = workspace / "harbor" / "benchmark.lock.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectiveError(
            f"run {run_id!r}: cannot read compiled lock at {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ObjectiveError(f"run {run_id!r}: compiled lock at {path} must be a mapping")
    return data


# The integer count keys a scored task must carry as JSON integers (mirrors
# the generated reward-row shape consumed by ``verifier_core.aggregate_metrics``).
_SCORED_COUNT_KEYS = ("tp", "fp", "fn")


def _parse_task_rows(
    job_dir: Path, run_id: str
) -> tuple[list[dict[str, object] | None], int]:
    """Read the run's per-task reward rows in strict fail-closed fashion.

    Walks the job dir exactly as ``run_mod._parse_job_results`` does (sorted
    trial subdirectories with ``<trial>/verifier/``): a scored task has a
    ``reward.json`` (parsed as a strict reward dict); a trial with only
    ``reward-details.json`` is an unscored infra failure and becomes a ``None``
    row (never a numeric zero).

    A task with ``clean_task == 1`` and zero findings/candidates is a
    legitimate clean task — it stays a scored row, never an infra failure. Any
    malformed ``reward.json`` (bad JSON, non-object, a non-integer count, any
    non-finite/negative value) raises ``ObjectiveError`` naming the artifact
    and run id.
    """
    if not job_dir.is_dir():
        return [], 0
    rows: list[dict[str, object] | None] = []
    infra_errors = 0
    for trial in sorted(job_dir.iterdir()):
        if not trial.is_dir():
            continue  # a non-directory sibling is not a task trial
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
    """Parse one ``reward.json`` into a strict reward dict.

    Fails closed on any malformed artifact: non-object JSON, a non-integer
    ``tp``/``fp``/``fn``/``verifier_error``, a negative or non-finite count, or
    a non-numeric ``reward``. The error message always names the artifact path
    and run id. Never ``unwrap_or`` a plausible default or coerce a malformed
    numeric to zero.
    """
    try:
        data = json.loads(reward_path.read_text(encoding="utf-8"))
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
        if int(value) < 0:
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


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _build_objective(
    rows: list[dict[str, object] | None], infra_errors: int,
    job_dir: Path, run_id: str,
) -> Objective:
    """Populate the count-derived ``Objective`` from the parsed rows.

    TP/FP/FN, precision/recall/f1 and the clean/task/scored/infra counts come
    from feeding the rows (infra rows included) through
    ``verifier_core.aggregate_metrics`` — the same authoritative pool the
    generated corpus metrics use. The verifier-error count is derived from the
    rows' ``verifier_error`` flags. ``comparison_eligible`` is ``False``
    whenever any trial failed to produce a legitimate scored row.
    """
    try:
        agg = verifier_core.aggregate_metrics(rows)
    except verifier_core.VerifierError as exc:
        raise ObjectiveError(
            f"malformed reward row(s) in run {run_id!r} under {job_dir}: {exc}"
        ) from exc
    verifier_errors = sum(
        1 for row in rows if row is not None and int(row.get("verifier_error") or 0) == 1
    )
    # Malformed rows fail closed in ``_parse_reward_strict`` and so never reach
    # this pool; the count stays zero by construction.
    malformed = 0
    failed = max(verifier_errors, infra_errors)
    candidate_count = sum(
        int(row["candidate_count"]) for row in rows if row is not None
    )
    gold_count = sum(int(row["gold_count"]) for row in rows if row is not None)
    clean_task_count = int(agg["clean_task_count"])
    return Objective(
        tp=int(agg["total_tp"]),
        fp=int(agg["total_fp"]),
        fn=int(agg["total_fn"]),
        precision=float(agg["micro_precision"]),
        recall=float(agg["micro_recall"]),
        f1=float(agg["micro_f1"]),
        clean_task_count=clean_task_count,
        clean_pass_count=int(round(agg["clean_accuracy"] * clean_task_count)),
        clean_accuracy=float(agg["clean_accuracy"]),
        task_count=int(agg["task_count"]),
        scored_task_count=int(agg["scored_task_count"]),
        candidate_count=candidate_count,
        gold_count=gold_count,
        infra_error_task_count=int(agg["infra_error_task_count"]),
        verifier_error_task_count=verifier_errors,
        malformed_task_count=malformed,
        failed_task_count=failed,
        comparison_eligible=not (infra_errors + verifier_errors + malformed + failed),
        mean_task_score=float(agg["mean_task_score"]),
    )

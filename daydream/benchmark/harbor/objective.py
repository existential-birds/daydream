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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import run as run_mod
from daydream.benchmark.harbor import verifier_core


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


@dataclass(frozen=True)
class CompletedRun:
    """Immutable projection of a completed, ledgered benchmark run."""

    run_id: str
    mode: str
    state: str
    # Bound in Task 3 (full compatibility identity).
    identity: Any | None = None
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
    whose ``state == "complete"``, then parses its per-task reward rows in
    strict fail-closed fashion and binds the count-derived ``Objective``.
    Missing runs, running/cleanup-pending/cleaned runs, any ledger parse/
    validation failure, and any malformed reward artifact all fail closed with
    an ``ObjectiveError`` naming the run id and the offending artifact.
    """
    del env  # reserved for provenance binding in later tasks.
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
    rows, infra_errors = _parse_task_rows(Path(job_dir), run_id)
    objective = _build_objective(rows, infra_errors)

    return CompletedRun(
        run_id=entry["run_id"],
        mode=entry["mode"],
        state=entry["state"],
        task_rows=rows,
        objective=objective,
        job_dir=job_dir,
    )


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
    rows: list[dict[str, object] | None], infra_errors: int
) -> Objective:
    """Populate the count-derived ``Objective`` from the parsed rows.

    TP/FP/FN, precision/recall/f1 and the clean/task/scored/infra counts come
    from feeding the rows (infra rows included) through
    ``verifier_core.aggregate_metrics`` — the same authoritative pool the
    generated corpus metrics use. The verifier-error count is derived from the
    rows' ``verifier_error`` flags. ``comparison_eligible`` is ``False``
    whenever any trial failed to produce a legitimate scored row.
    """
    agg = verifier_core.aggregate_metrics(rows)
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

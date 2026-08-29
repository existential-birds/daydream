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

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

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
    location_pairs_scored: int = 0
    severity_pairs_scored: int = 0
    location_exact: int = 0
    location_near: int = 0
    location_file: int = 0
    location_miss: int = 0
    total_location_exact: int = 0
    total_location_near: int = 0
    total_location_file: int = 0
    total_location_miss: int = 0
    severity_exact: int = 0
    severity_within_1: int = 0
    total_severity_exact: int = 0
    total_severity_within_1: int = 0
    location_exact_rate: float = 0.0
    location_near_rate: float = 0.0
    location_file_rate: float = 0.0
    location_miss_rate: float = 0.0
    severity_exact_rate: float = 0.0
    severity_within_1_rate: float = 0.0
    severity_mean_distance: float = 0.0
    severity_credit: float = 0.0
    location_credit: float = 0.0
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
            "location_pairs_scored": self.location_pairs_scored,
            "severity_pairs_scored": self.severity_pairs_scored,
            "location_exact_rate": self.location_exact_rate,
            "location_near_rate": self.location_near_rate,
            "location_file_rate": self.location_file_rate,
            "location_miss_rate": self.location_miss_rate,
            "location_credit": self.location_credit,
            "location_exact": self.location_exact,
            "location_near": self.location_near,
            "location_file": self.location_file,
            "location_miss": self.location_miss,
            "total_location_exact": self.total_location_exact,
            "total_location_near": self.total_location_near,
            "total_location_file": self.total_location_file,
            "total_location_miss": self.total_location_miss,
            "severity_exact": self.severity_exact,
            "severity_within_1": self.severity_within_1,
            "total_severity_exact": self.total_severity_exact,
            "total_severity_within_1": self.total_severity_within_1,
            "severity_exact_rate": self.severity_exact_rate,
            "severity_within_1_rate": self.severity_within_1_rate,
            "severity_mean_distance": self.severity_mean_distance,
            "severity_credit": self.severity_credit,
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
class SuiteEntry:
    """One exact completion referenced by a suite manifest.

    ``workspace`` is the repository workspace path the run was recorded against
    and ``run_id`` resolves the exact ledgered completion within it. Instances
    are immutable and preserve manifest order.
    """

    workspace: Path
    run_id: str


def identity_to_dict(identity: CompatibilityIdentity) -> dict[str, object]:
    """Canonical 18-field compatibility/identity projection.

    Single source of truth for the identity mapping reused by
    ``objective_to_json``, the suite aggregate identity, and the CLI's
    ``_suite_objective_to_json`` (issue #888 anti-slop: adding/renaming a field
    in one place must not silently desynchronize the others). Repository/
    benchmark ids are deliberately not part of the identity.
    """
    return {
        "objective_schema_version": identity.objective_schema_version,
        "profile_schema_version": identity.profile_schema_version,
        "profile_name": identity.profile_name,
        "profile_digest": identity.profile_digest,
        "daydream_version": identity.daydream_version,
        "daydream_wheel_sha256": identity.daydream_wheel_sha256,
        "compiled_lock_sha256": identity.compiled_lock_sha256,
        "harbor_version": identity.harbor_version,
        "reviewer_backend": identity.reviewer_backend,
        "reviewer_model": identity.reviewer_model,
        "reviewer_base_url": identity.reviewer_base_url,
        "reviewer_effort": identity.reviewer_effort,
        "judge_provider": identity.judge_provider,
        "judge_model": identity.judge_model,
        "judge_host": identity.judge_host,
        "verifier_template_sha256": identity.verifier_template_sha256,
        "threshold": identity.threshold,
        "attempts": identity.attempts,
    }


@dataclass(frozen=True)
class SuiteObjective:
    """A pooled, compatible suite of exact completions.

    ``objective`` holds the count-derived micro-metrics pooled across every
    entry's flattened per-task rows (never per-repository averages).
    ``experiment_id`` is a stable SHA-256 derived from the canonicalized
    manifest plus the shared compatibility identity. ``identity`` is the
    (single, verified-shared) ``CompatibilityIdentity``; ``profile_digest`` is
    always present (spec must-have). ``diagnostics`` carries per-entry
    ``{index, workspace, run_id, error}`` records for the error/reporting path
    -- never prose-only -- and is empty on a cleanly pooled suite.
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
        job_dir=job_dir,
    )


# The schema version recorded in ``CompatibilityIdentity.objective_schema_version``.
_OBJECTIVE_SCHEMA_VERSION = 1


def objective_to_json(run: CompletedRun) -> dict[str, object]:
    """Project a completed run into opaque, privacy-safe machine-readable JSON.

    Produces only the opaque ``run_id``, ``mode``, ``schema_version``, the
    ``identity`` dict, and the ``objective`` dict (counts + ``comparison_eligible``
    + the reported location/severity axes + optional ``tokens``/``cost``). No
    repository slug, PR number, source path,
    gold/candidate text, judge reasoning, or source code is ever emitted; only
    opaque benchmark/run ids and counts pass through (spec privacy must-have).

    The ledger entry's ``job_dir`` is deliberately dropped before projection so
    the workspace filesystem path never leaks into the output.
    """
    identity = run.identity
    identity_json = None
    if identity is not None:
        identity_json = identity_to_dict(identity)

    objective_dict: dict[str, object] | None = None
    if run.objective is not None:
        obj = run.objective
        objective_dict = {
            "tp": obj.tp,
            "fp": obj.fp,
            "fn": obj.fn,
            "precision": obj.precision,
            "recall": obj.recall,
            "f1": obj.f1,
            "clean_task_count": obj.clean_task_count,
            "clean_pass_count": obj.clean_pass_count,
            "clean_accuracy": obj.clean_accuracy,
            "task_count": obj.task_count,
            "scored_task_count": obj.scored_task_count,
            "candidate_count": obj.candidate_count,
            "gold_count": obj.gold_count,
            "infra_error_task_count": obj.infra_error_task_count,
            "verifier_error_task_count": obj.verifier_error_task_count,
            "malformed_task_count": obj.malformed_task_count,
            "failed_task_count": obj.failed_task_count,
            "comparison_eligible": obj.comparison_eligible,
            "mean_task_score": obj.mean_task_score,
            "location_pairs_scored": obj.location_pairs_scored,
            "severity_pairs_scored": obj.severity_pairs_scored,
            "location_exact_rate": obj.location_exact_rate,
            "location_near_rate": obj.location_near_rate,
            "location_file_rate": obj.location_file_rate,
            "location_miss_rate": obj.location_miss_rate,
            "location_credit": obj.location_credit,
            "location_exact": obj.location_exact,
            "location_near": obj.location_near,
            "location_file": obj.location_file,
            "location_miss": obj.location_miss,
            "total_location_exact": obj.total_location_exact,
            "total_location_near": obj.total_location_near,
            "total_location_file": obj.total_location_file,
            "total_location_miss": obj.total_location_miss,
            "severity_exact": obj.severity_exact,
            "severity_within_1": obj.severity_within_1,
            "total_severity_exact": obj.total_severity_exact,
            "total_severity_within_1": obj.total_severity_within_1,
            "severity_exact_rate": obj.severity_exact_rate,
            "severity_within_1_rate": obj.severity_within_1_rate,
            "severity_mean_distance": obj.severity_mean_distance,
            "severity_credit": obj.severity_credit,
        }
        if obj.tokens is not None:
            objective_dict["tokens"] = obj.tokens
        if obj.cost is not None:
            objective_dict["cost"] = obj.cost

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
    """Stable SHA-256 over the canonicalized manifest plus the shared identity.

    Canonicalizing (sorting unique entries) means reordering identical unique
    entries yields the same id; duplicates are rejected before this runs.
    """
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
    """Validate and pool a suite manifest into one compatible ``SuiteObjective``.

    Validates the manifest, resolves every entry via ``read_completed_run``, and
    requires the full compatibility identity to match across every entry
    (non-optional): any differing compatibility field — profile digest, wheel/
    runtime digest, reviewer/judge identity, verifier template, threshold, or
    attempts — raises ``ObjectiveError`` naming the field. Any entry that is
    incomplete, malformed, comparison-ineligible, or a duplicated pair fails the
    entire command fail-closed — never a silently-subsetted pool.

    The pooled objective feeds the flattened per-task rows across all entries
    through ``verifier_core.aggregate_metrics`` exactly once; TP/FP/FN pool to
    micro precision/recall/F1 and the task/clean/infra counts sum across entries.
    Per-repository precision/recall/F1 are never averaged.
    """
    env = env or {}
    entries = validate_suite_manifest(manifest)

    resolved: list[tuple[SuiteEntry, CompletedRun]] = []
    diagnostics: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        try:
            run = read_completed_run(entry.workspace, entry.run_id, env=env)
        except ObjectiveError as exc:
            diagnostics.append({
                "index": index,
                "workspace": str(entry.workspace),
                "run_id": entry.run_id,
                "error": str(exc),
            })
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
        diagnostics=diagnostics,
    )


def _suite_label(entries: list[SuiteEntry]) -> str:
    return "-".join(f"{e.workspace.name}:{e.run_id}" for e in entries)


_PROFILE_SCHEMA_VERSION = 1

_SUITE_SCHEMA_VERSION = 1


def validate_suite_manifest(manifest: dict[str, Any]) -> list[SuiteEntry]:
    """Validate a suite manifest and return its entries in manifest order.

    Rejects a manifest whose ``schema_version`` is not 1, a missing/non-list
    ``entries``, a non-dict entry, or an entry missing ``workspace``/``run_id``
    --- every failure raises ``ObjectiveError``. Any duplicated
    ``(workspace, run_id)`` pair is also rejected, naming the offending entry
    and its index. Bad entries are never skipped nor coerced to a default;
    the error always carries the entry index and offending field.
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
    """Parse one ``reward.json`` into a strict reward dict.

    Fails closed on any malformed artifact: non-object JSON, a non-integer
    ``tp``/``fp``/``fn``/``verifier_error``, a negative or non-finite count, or
    a non-numeric ``reward``. The error message always names the artifact path
    and run id. Never ``unwrap_or`` a plausible default or coerce a malformed
    numeric to zero.
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
    """Read an integer-valued row field, defaulting to 0 when absent.

    Only the scored count keys are validated strictly at parse time; auxiliary
    reporting keys (``candidate_count``/``gold_count``/``verifier_error``) are
    read here with a non-integer value falling back to 0 rather than crashing.
    """
    value = row.get(key)
    return value if _is_int(value) else 0


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
        comparison_eligible=bool(agg["scored_task_count"]) and not (
            infra_errors + verifier_errors + malformed + failed
        ),
        mean_task_score=float(agg["mean_task_score"]),
        location_pairs_scored=int(agg["location_pairs_scored"]),
        severity_pairs_scored=int(agg["severity_pairs_scored"]),
        location_exact_rate=float(agg["location_exact_rate"]),
        location_near_rate=float(agg["location_near_rate"]),
        location_file_rate=float(agg["location_file_rate"]),
        location_miss_rate=float(agg["location_miss_rate"]),
        location_credit=float(agg["location_credit"]),
        location_exact=int(agg["location_exact"]),
        location_near=int(agg["location_near"]),
        location_file=int(agg["location_file"]),
        location_miss=int(agg["location_miss"]),
        total_location_exact=int(agg["total_location_exact"]),
        total_location_near=int(agg["total_location_near"]),
        total_location_file=int(agg["total_location_file"]),
        total_location_miss=int(agg["total_location_miss"]),
        severity_exact=int(agg["severity_exact"]),
        severity_within_1=int(agg["severity_within_1"]),
        total_severity_exact=int(agg["total_severity_exact"]),
        total_severity_within_1=int(agg["total_severity_within_1"]),
        severity_exact_rate=float(agg["severity_exact_rate"]),
        severity_within_1_rate=float(agg["severity_within_1_rate"]),
        severity_mean_distance=float(agg["severity_mean_distance"]),
        severity_credit=float(agg["severity_credit"]),
    )

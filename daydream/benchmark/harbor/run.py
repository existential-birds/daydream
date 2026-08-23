"""Supervised Harbor runs behind the Oracle self-match gate (issue #781).

A thin safety wrapper around Harbor 0.21 that fail-closes on every preflight
before Harbor starts (same-interpreter Harbor, compiled-tree presence,
endpoint hosts vs the workspace allowlists, telemetry/upload rejection,
Docker allowlist support, and — for the Oracle pass — a current
``runtime/calibration-receipt.json``), prints a pre-run spend summary, and
records every run in a private ``runtime/harbor.json`` cleanup ledger. Harbor
remains the only orchestrator/results implementation; this module only
selects the already-compiled config, drives ``harbor run -c <config>`` with
the process CWD set to the absolute ``<workspace>/harbor`` directory, and
parses the spike-confirmed job/result layout (jobs/<job>/<trial>/verifier/
reward.json) afterwards. The whole run path is driven through injectable
seams (``spawn`` / ``docker_ok`` / ``confirm``) so CI stays hermetic.

Expected failures are a ``RunError`` family with a clear message; the CLI
handler maps them to exit ``1`` — never a bare traceback.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import yaml

from daydream.benchmark import storage
from daydream.benchmark.harbor import calibrate, package


class RunError(Exception):
    """Base error for the supervised benchmark run subsystem."""


class RunBlocked(RunError):
    """A preflight/gate refused to let the run start (fail-closed)."""


def _load_workspace_privacy(workspace: Path) -> dict[str, Any]:
    """Return the raw ``benchmark.yaml`` privacy block.

    Falls back to the raw manifest dict (never the strict ``Privacy`` model)
    so a privacy field like ``uploads: enabled`` is *reported* by the preflight
    rather than rejected wholesale as corrupt. A malformed/missing manifest
    raises the project's existing ``WorkspaceCorrupt`` — never a silent default.
    """
    try:
        raw = storage.load_yaml_strict(workspace / "benchmark.yaml")
    except storage.WorkspaceCorrupt:
        raise
    privacy = raw.get("privacy")
    if not isinstance(privacy, dict):
        raise storage.WorkspaceCorrupt(
            f"{workspace}: benchmark.yaml is missing a privacy block"
        )
    return privacy


def _normalize_allowlist(values: Any, what: str) -> list[str]:
    """Normalize a workspace allowlist with the schema's hostname rules."""
    from daydream.benchmark.schema import normalize_hostname

    if not isinstance(values, list) or not values:
        raise RunBlocked(f"privacy {what} must be a non-empty host list")
    return [normalize_hostname(str(v)) for v in values]


def _judge_host_from_env(env: dict[str, Any]) -> str:
    """Judge egress host for ``env`` (mirrors calibrate's resolution)."""
    return calibrate._judge_host_from_env(env)


def _reviewer_host_from_env(env: dict[str, Any]) -> str:
    """Reviewer egress host for ``env``: base-URL host, else the anthropic default.

    Mirrors ``calibrate._judge_host_from_env``: a configured
    ``DAYDREAM_REVIEW_BASE_URL`` resolves to its hostname (lowercased, no
    port); without one the reviewer routes to the default ``api.anthropic.com``.
    """
    base = env.get("DAYDREAM_REVIEW_BASE_URL") or ""
    if not base:
        return "api.anthropic.com"
    return str(urllib.parse.urlsplit(base).hostname or "").lower()


def _compiled_job_config(workspace: Path) -> dict[str, Any]:
    """Read the compiled ``harbor/harbor-job.yaml`` config as a dict.

    Fallible: ``OSError``/``YAML`` errors surface a ``RunError`` (a malformed
    config is a block, not a best-effort summary).
    """
    path = workspace / "harbor" / "harbor-job.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RunError(f"cannot parse compiled Harbor job config at {path}: {exc}") from exc
    except OSError as exc:
        raise RunError(f"cannot read compiled Harbor job config at {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RunError(f"compiled Harbor job config at {path} must be a mapping")
    return data


def _compiled_cases(workspace: Path) -> list[dict[str, Any]]:
    """Return the compiled lock's case entries (defensive over dict/list)."""
    path = workspace / "harbor" / "benchmark.lock.json"
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"cannot read compiled lock at {path}: {exc}") from exc
    cases = lock.get("cases") if isinstance(lock, dict) else None
    if isinstance(cases, dict):
        return list(cases.values())
    if isinstance(cases, list):
        return list(cases)
    return []


def _pre_run_summary(workspace: Path, *, env: dict[str, Any]) -> str:
    """Human-readable pre-run spend summary over validated inputs only.

    Pure string-building (never reads Harbor output / never makes a network
    call). Missing env values render as ``unset``; a broken lock/config raises
    ``RunError``.
    """
    config = _compiled_job_config(workspace)
    cases = _compiled_cases(workspace)

    gold_candidate_pairs = sum(
        1 for c in cases if isinstance(c, dict)
    )
    oracle_pairs = sum(
        int(c.get("oracle_count", 0) or 0) for c in cases if isinstance(c, dict)
    )

    attempts = config.get("n_attempts")
    concurrency = config.get("n_concurrent_trials")
    first_case = cases[0] if cases else {}
    timeout_sec = None
    if isinstance(first_case, dict):
        timeout_sec = first_case.get("timeout_sec")
    if timeout_sec is None:
        timeout_sec = config.get("timeout_sec")

    def _or(value: Any, default: str = "unset") -> str:
        return default if value is None else str(value)

    judge_host = _judge_host_from_env(env)
    return "\n".join(
        [
            "Pre-run Harbor spend summary",
            f"  task count:       {len(cases)}",
            f"  reviewer model:   {env.get('DAYDREAM_REVIEW_MODEL', '') or 'unset'}",
            f"  judge provider:   {env.get('DAYDREAM_JUDGE_PROVIDER', '') or 'unset'}",
            f"  judge model:      {env.get('DAYDREAM_JUDGE_MODEL', '') or 'unset'}",
            f"  judge host:       {judge_host or 'unset'}",
            f"  attempts:         {_or(attempts)}",
            f"  concurrency:      {_or(concurrency)}",
            f"  timeouts:         {_or(timeout_sec)}",
            f"  oracle pair:      {_or(oracle_pairs, '0')}",
            f"  benchmark judge pair: {_or(gold_candidate_pairs, '0')}",
            "reviewer spend is time-bounded (a per-turn timeout), not a strict dollar cap",
        ]
    )


def _compiled_lock_sha256(workspace: Path) -> str:
    """sha256 of the compiled ``harbor/benchmark.lock.json`` bytes."""
    return hashlib.sha256((workspace / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()


LEDGER_SUPPORTED_STATES = ("running", "complete", "cleanup_pending", "cleaned")
LEDGER_SUPPORTED_MODES = ("oracle", "benchmark")
LEDGER_SUPPORTED_BACKENDS = ("docker",)


def _ledger_path(workspace: Path) -> Path:
    return workspace / "runtime" / "harbor.json"


def _load_ledger(workspace: Path) -> dict[str, Any]:
    """Read ``runtime/harbor.json``; initialise a fresh ledger when absent.

    A malformed existing entry (bad JSON / missing keys / unsupported backend /
    environment ref without an exact ``image_id``) raises ``RunError`` —
    corruption is never silently dropped or broadened.
    """
    path = _ledger_path(workspace)
    if not path.is_file():
        return {"schema_version": 1, "runs": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"malformed cleanup ledger at {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise RunError(f"unrecognised cleanup ledger schema at {path}")
    runs = doc.get("runs")
    if not isinstance(runs, list):
        raise RunError(f"cleanup ledger at {path} is missing its runs list")
    for run in runs:
        _validate_ledger_entry(path, run)
    return doc


def _validate_ledger_entry(path: Path, run: Any) -> None:
    if not isinstance(run, dict):
        raise RunError(f"ledger {path} has a non-object run entry")
    for key in ("run_id", "mode", "state", "compiled_lock_sha256", "job_dir"):
        if key not in run:
            raise RunError(f"ledger {path} run entry missing {key!r}")
    if run.get("mode") not in LEDGER_SUPPORTED_MODES:
        raise RunError(f"ledger {path} run entry has unsupported mode {run.get('mode')!r}")
    if run.get("state") not in LEDGER_SUPPORTED_STATES:
        raise RunError(f"ledger {path} run entry has unsupported state {run.get('state')!r}")
    for env in run.get("environments") or []:
        if not isinstance(env, dict):
            raise RunError(f"ledger {path} run entry has a non-object environment")
        if env.get("backend") not in LEDGER_SUPPORTED_BACKENDS:
            raise RunError(
                f"ledger {path} environment has unsupported backend {env.get('backend')!r}"
            )
        if not env.get("image_id"):
            raise RunError(f"ledger {path} environment ref lacks an exact image_id")


def _validate_job_dir(workspace: Path, job_dir: str) -> str:
    """Reject a job dir that is not contained under ``<ws>/harbor/jobs/``."""
    root = Path(job_dir).expanduser().resolve() if job_dir else Path(job_dir)
    jobs_root = (workspace / "harbor" / "jobs").resolve()
    if not root.is_absolute() or not root.is_relative_to(jobs_root):
        raise RunError(f"run job dir must live under {jobs_root}, got {job_dir!r}")
    return str(root)


def ledger_append_running(
    workspace: Path, *, run_id: str, compiled_lock_sha256: str, job_dir: str,
    mode: str = "oracle",
) -> None:
    """Append a ``running`` entry for a unique job dir (block-before-spawn)."""
    contained = _validate_job_dir(workspace, job_dir)
    with storage.WorkspaceLock(workspace):
        doc = _load_ledger(workspace)
        entry = {
            "run_id": run_id,
            "mode": mode,
            "state": "running",
            "compiled_lock_sha256": compiled_lock_sha256,
            "job_dir": contained,
            "harbor_job_id": None,
            "environments": [],
            "error": None,
        }
        doc["runs"].append(entry)
        storage.atomic_write_json(_ledger_path(workspace), doc, mode=0o600)


def ledger_mark(
    workspace: Path, run_id: str, *, state: str,
    environments: list[dict[str, Any]] | None = None,
    harbor_job_id: str | None = None, error: str | None = None,
) -> None:
    """Update an existing ledger entry's terminal state + optional fields."""
    path = _ledger_path(workspace)
    with storage.WorkspaceLock(workspace):
        doc = _load_ledger(workspace)
        for run in doc["runs"]:
            if run.get("run_id") == run_id:
                run["state"] = state
                if environments is not None:
                    run["environments"] = environments
                    _validate_ledger_entry(path, run)
                if harbor_job_id is not None:
                    run["harbor_job_id"] = harbor_job_id
                if error is not None:
                    run["error"] = error
                storage.atomic_write_json(path, doc, mode=0o600)
                return
        raise RunError(f"run {run_id!r} not found in cleanup ledger {path}")


def _calibration_invalidation_inputs(env: dict[str, Any]) -> dict[str, Any]:
    """The calibration-receipt invalidation inputs for ``env`` (reused verbatim)."""
    sr = calibrate._load_judge_template()
    pairs = calibrate._load_fixture()
    return calibrate._invalidation_inputs(env, pairs, sr)


def _preflight(
    workspace: Path,
    *,
    oracle: bool,
    env: dict[str, Any],
    docker_ok: Callable[[], bool],
) -> list[str]:
    """Fail-closed preflight: collect every blocking failure (empty = pass).

    Checks, in order, without stopping at the first failure:
      1. same-interpreter Harbor resolution + compiled-tree presence
      2. judge/reviewer egress hosts vs the workspace allowlists
      3. telemetry/upload rejection (archive + uploads must be ``disabled``)
      4. Docker allowlist support (no public-networking fallback)
      5. (oracle only) a current passed calibration receipt
    """
    failures: list[str] = []

    try:
        package.resolve_harbor()
    except package.PackageError as exc:
        failures.append(str(exc))
    compiled = workspace / "harbor"
    config_name = "harbor-oracle.yaml" if oracle else "harbor-job.yaml"
    for required in ("benchmark.lock.json", config_name):
        if not (compiled / required).is_file():
            failures.append(f"missing compiled Harbor tree file: harbor/{required}")

    try:
        privacy = _load_workspace_privacy(workspace)
    except storage.WorkspaceCorrupt as exc:
        failures.append(str(exc))
        privacy = {}

    if privacy:
        try:
            reviewer_hosts = _normalize_allowlist(
                privacy.get("reviewer_allowed_hosts"), "reviewer_allowed_hosts"
            )
            judge_hosts = _normalize_allowlist(
                privacy.get("judge_allowed_hosts"), "judge_allowed_hosts"
            )
        except RunBlocked as exc:
            failures.append(str(exc))
            reviewer_hosts, judge_hosts = [], []
        try:
            judge_host = _judge_host_from_env(env)
        except Exception as exc:  # noqa: BLE001 - surfaced as a preflight failure
            failures.append(f"cannot resolve judge host: {exc}")
            judge_host = ""
        try:
            reviewer_host = _reviewer_host_from_env(env)
        except Exception as exc:  # noqa: BLE001 - surfaced as a preflight failure
            failures.append(f"cannot resolve reviewer host: {exc}")
            reviewer_host = ""
        if judge_host and judge_host not in judge_hosts:
            failures.append(
                f"judge host {judge_host!r} is not in the workspace judge_allowed_hosts allowlist"
                f" ({sorted(judge_hosts)})"
            )
        if reviewer_host and reviewer_host not in reviewer_hosts:
            failures.append(
                f"reviewer host {reviewer_host!r} is not in the workspace "
                f"reviewer_allowed_hosts allowlist ({sorted(reviewer_hosts)})"
            )
        for field in ("archive", "uploads"):
            value = privacy.get(field)
            if value != "disabled":
                failures.append(
                    f"privacy {field} must be disabled before a Harbor run (got {value!r})"
                )

    if not docker_ok():
        failures.append(
            "Docker allowlist is unsupported on the selected runtime; "
            "refusing to fall back to public networking"
        )

    if oracle:
        receipt_path = workspace / "runtime" / "calibration-receipt.json"
        try:
            current = calibrate.is_receipt_current(
                receipt_path, _calibration_invalidation_inputs(env)
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a preflight failure
            current = False
            failures.append(f"calibration receipt check failed: {exc}")
        if not current:
            failures.append(
                "no current calibration receipt at runtime/calibration-receipt.json "
                "(run 'daydream benchmark calibrate-judge' first)"
            )
    return failures


def run_run(
    workspace: Path,
    *,
    oracle: bool = False,
    yes: bool = False,
    env: dict[str, Any] | None = None,
    spawn=None,
    docker_ok=None,
    confirm=None,
) -> int:
    """Supervised Harbor run — wired end-to-end in the orchestrator task."""
    raise NotImplementedError("run_run lands with the orchestrator task")

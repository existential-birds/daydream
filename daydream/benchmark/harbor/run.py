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

import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable

import yaml

from daydream.benchmark import storage
from daydream.benchmark.harbor import calibrate, package, verifier_core


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

    attempts = config.get("n_attempts")
    concurrency = config.get("n_concurrent_trials")
    # The oracle judges every compiled case once per attempt, so the oracle
    # spend is the case count times the configured retries.
    oracle_pairs = len([c for c in cases if isinstance(c, dict)]) * int(attempts or 1)
    first_case = cases[0] if cases else {}
    timeout_sec = None
    if isinstance(first_case, dict):
        timeout_sec = first_case.get("timeout_sec")
    if timeout_sec is None:
        timeout_sec = config.get("timeout_sec")

    def _or(value: Any, default: str = "unset") -> str:
        return default if value is None else str(value)

    judge_host = calibrate._judge_host_from_env(env)
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


def _compiled_daydream_wheel(workspace: Path) -> tuple[str, str]:
    """The ``daydream`` wheel ``(version, sha256)`` from the compiled lock.

    Authoritative provenance source (issue #888): the compiled
    ``harbor/benchmark.lock.json`` ``daydream`` block describing the exact
    Daydream wheel the run executed under. Fail-closed: an absent/malformed
    block (or unreadable lock) raises ``RunError`` naming the lock path — a
    plausible placeholder is never defaulted.
    """
    path = workspace / "harbor" / "benchmark.lock.json"
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"cannot read compiled lock at {path}: {exc}") from exc
    if not isinstance(lock, dict):
        raise RunError(f"compiled lock at {path} must be a mapping")
    day = lock.get("daydream")
    if not isinstance(day, dict) or not isinstance(day.get("version"), str) \
            or not isinstance(day.get("sha256"), str):
        raise RunError(
            f"compiled lock at {path} is missing its 'daydream' wheel block "
            "(version/sha256)"
        )
    return day["version"], day["sha256"]


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
    profile_digest: str | None = None,
    reviewer_effort: str | None = None,
) -> None:
    """Append a ``running`` entry (written before Harbor spawns) for this run.

    Uniqueness is the freshly generated uuid4 ``run_id``; the ``job_dir`` is
    validated only for containment under ``<ws>/harbor/jobs/``.

    ``profile_digest`` (issue #885/R12) is the canonical digest of the
    review-profile candidate this run executes under, supplied by the control
    plane (the entrypoint computes it from the validated candidate); this
    function never reads ambient env itself. Optional: legacy/late callers
    that omit it leave the field ``None`` on the entry.

    ``reviewer_effort`` (issue #888) is the reviewer reasoning-effort this run
    executed under, threaded from the control-plane env so the read-only
    objective can recover it without inference. Optional: callers that omit it
    leave the field ``None`` on the entry (legacy/late callers stay byte-stable).
    """
    validated = _validate_job_dir(workspace, job_dir)
    with storage.WorkspaceLock(workspace):
        doc = _load_ledger(workspace)
        entry: dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "state": "running",
            "compiled_lock_sha256": compiled_lock_sha256,
            "job_dir": validated,
            "harbor_job_id": None,
            "environments": [],
            "error": None,
            "profile_digest": profile_digest,
            "reviewer_effort": reviewer_effort,
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
    inputs = calibrate._invalidation_inputs(env, pairs, sr)
    # Candidate review-profile digest (issue #885/R12): when the control plane
    # supplies a ``DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST``, a change of
    # candidate invalidates the calibration receipt (benchmark attribution
    # requires per-candidate invalidation). Omitted when absent so the
    # legacy receipt contract stays byte-stable for default-profile runs.
    digest = env.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST")
    if digest:
        inputs["profile_digest"] = str(digest)
    return inputs


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
        for label, resolve, hosts in (
            ("judge", calibrate._judge_host_from_env, judge_hosts),
            ("reviewer", _reviewer_host_from_env, reviewer_hosts),
        ):
            try:
                host = resolve(env)
            except Exception as exc:  # noqa: BLE001 - surfaced as a preflight failure
                failures.append(f"cannot resolve {label} host: {exc}")
                continue
            if host and host not in hosts:
                failures.append(
                    f"{label} host {host!r} is not in the workspace "
                    f"{label}_allowed_hosts allowlist ({sorted(hosts)})"
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


def _iter_trial_dirs(job_dir: Path):
    """Yield the sorted trial subdirectories of a Harbor job dir.

    Non-directory siblings (lockfiles, READMEs) are skipped. Shared by
    ``_parse_job_results`` (oracle path) and ``objective._parse_task_rows`` so
    the trial-dir traversal skeleton lives in one place (issue #888 anti-slop).
    """
    for trial in sorted(job_dir.iterdir()):
        if trial.is_dir():
            yield trial


def _parse_job_results(job_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Parse Harbor's job dir for per-task scores + resolved environments.

    Fail-closed: returns ``(oracle_ok, environments)`` where a trial with no
    score evidence (neither ``reward.json`` nor ``reward-details.json`` — e.g.
    Harbor returned/wrote nothing) or an empty job dir blocks the oracle rather
    than passing with zero per-task evidence. A task is scored when its
    ``<trial>/verifier/reward.json`` exists; it is *unscored* when only
    ``reward-details.json`` exists (the infra-error path — never a numeric
    zero). Oracle success requires every trial scored with ``reward == 1.0``
    and ``verifier_error == 0``.
    """
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        return (False, [])
    environments: list[dict[str, Any]] = []
    oracle_ok = True
    for trial in _iter_trial_dirs(job_dir):
        verifier = trial / "verifier"
        reward_path = verifier / "reward.json"
        # Resolve the trial environment even when the trial carries no claimable
        # score evidence, so a failed/aborted run still records the Docker
        # images it spawned: ``clean --jobs`` can address them rather than
        # deleting the job dir and permanently stranding the images.
        if not reward_path.is_file():
            return (False, [*environments, _environment_from_trial(trial)])
        env = _environment_from_trial(trial)
        reward = _parse_reward(reward_path)
        if reward is None:
            return (False, [*environments, env])
        environments.append(env)
        if not oracle_ok:
            continue
        verr = reward.get("verifier_error")
        if verr is None or float(reward.get("reward") or 0.0) < 1.0 or int(verr) != 0:
            oracle_ok = False
    # An empty job dir (Harbor returned and wrote nothing) is not a
    # reproduction: the oracle must present evidence for every compiled task.
    if not environments:
        return (False, [])
    return (oracle_ok, environments)


def _parse_reward(reward_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(reward_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _environment_from_trial(trial: Path) -> dict[str, Any]:
    """Deterministic resolved-environment entry for one scored trial.

    Computes ``environment_id`` as a content-address over the trial's compiled
    environment (task.toml ``[environment]`` + environment/ bytes); the docker
    provider tags the built image ``hb__<environment_id>`` when the task does
    not pin a concrete ``docker_image``.
    """
    digest = hashlib.sha256()
    task_toml = trial / "task.toml"
    if task_toml.is_file():
        digest.update(task_toml.name.encode("utf-8"))
        digest.update(task_toml.read_bytes())
    env_root = trial / "environment"
    if env_root.is_dir():
        for path in sorted(env_root.rglob("*")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    environment_id = digest.hexdigest()
    return {
        "trial_name": trial.name,
        "environment_id": environment_id,
        "backend": "docker",
        "image_id": f"hb__{environment_id}",
        "image_tags": [],
        "removed": False,
    }


def _current_state_mapping(
    *, workspace: Path, compiled_lock_sha256: str, env: dict[str, Any],
    calibration_digest: str,
) -> dict[str, Any]:
    """The current Oracle/Harbor state an oracle receipt must match.

    Shared verbatim by the receipt document (``_oracle_receipt_document``) and
    the default-run gate (``_default_run_gate``) so the two cannot drift.
    """
    version = importlib.metadata.version("harbor")
    major_minor = ".".join(str(version).split(".")[:2])
    sr = calibrate._load_judge_template()
    config = _compiled_job_config(workspace)
    mapping = {
        "compiled_lock_sha256": compiled_lock_sha256,
        "harbor_version": major_minor,
        "judge_provider": env.get("DAYDREAM_JUDGE_PROVIDER") or "anthropic",
        "judge_model": env.get("DAYDREAM_JUDGE_MODEL") or "",
        "judge_host": calibrate._judge_host_from_env(env),
        "reviewer_backend": env.get("DAYDREAM_REVIEW_BACKEND") or "",
        "reviewer_model": env.get("DAYDREAM_REVIEW_MODEL") or "",
        "reviewer_base_url": env.get("DAYDREAM_REVIEW_BASE_URL") or "",
        "verifier_template_sha256": calibrate._render_judge_prompt_digest(sr),
        "threshold": verifier_core.CONFIDENCE_THRESHOLD,
        "attempts": config.get("n_attempts", 1),
        "calibration_receipt_sha256": calibration_digest,
    }
    # Daydream wheel provenance (issue #888): bind the exact compiled wheel
    # digest/version the run was built under from the authoritative lock. An
    # absent/malformed block raises ``RunError`` naming the lock path — never a
    # default.
    wheel_version, wheel_sha = _compiled_daydream_wheel(workspace)
    mapping["daydream_version"] = wheel_version
    mapping["daydream_wheel_sha256"] = wheel_sha
    # Candidate review-profile digest (issue #885/R12): fold it into the shared
    # oracle state so both the oracle-receipt document and the default-run gate
    # compare the tested candidate. Omitted when no candidate is set so legacy
    # oracle receipts stay byte-stable.
    digest = env.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST")
    if digest:
        mapping["profile_digest"] = str(digest)
    # Reviewer reasoning-effort (issue #888): the control plane threads the
    # reviewer effort under which this run executes into the shared state via
    # the env, so both the oracle-receipt document and the default-run gate
    # compare the identical effort. Always present (``""`` when unset) so a
    # reviewer-less run's receipt and gate agree.
    mapping["reviewer_effort"] = env.get("DAYDREAM_REVIEW_EFFORT") or ""
    return mapping


def _oracle_receipt_document(
    *, workspace: Path, compiled_lock_sha256: str, env: dict[str, Any],
    calibration_digest: str, result_dir: Path,
) -> dict[str, Any]:
    """Assemble the private deterministic Oracle receipt (mode-0600)."""
    doc = _current_state_mapping(
        workspace=workspace, compiled_lock_sha256=compiled_lock_sha256, env=env,
        calibration_digest=calibration_digest,
    )
    doc["result_dir"] = str(Path(result_dir).resolve())
    doc["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return doc


def _write_oracle_receipt(
    workspace: Path, *, job_dir: Path, compiled_lock_sha256: str,
    env: dict[str, Any], calibration_digest: str,
) -> int:
    """Write the oracle-receipt only when the Oracle run reproduced gold."""
    ok, _ = _parse_job_results(Path(job_dir))
    if not ok:
        print(
            "Oracle run did not reproduce gold (a scored task/reward blocked the "
            "receipt); no oracle-receipt.json written.",
            file=sys.stderr,
        )
        return 1
    doc = _oracle_receipt_document(
        workspace=workspace, compiled_lock_sha256=compiled_lock_sha256, env=env,
        calibration_digest=calibration_digest, result_dir=Path(job_dir),
    )
    storage.atomic_write_json(
        workspace / "harbor" / "oracle-receipt.json", doc, mode=0o600
    )
    return 0


def _default_run_gate(
    workspace: Path, *, env: dict[str, Any], compiled_lock_sha256: str,
    calibration_digest: str,
) -> str | None:
    """Gate a default (non-Oracle) run behind a matching Oracle receipt."""
    receipt_path = workspace / "harbor" / "oracle-receipt.json"
    if not receipt_path.is_file():
        return "no matching oracle receipt found (run --oracle first)"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"malformed oracle receipt at {receipt_path}: {exc}"
    if not isinstance(receipt, dict):
        return f"malformed oracle receipt at {receipt_path}"
    current = _current_state_mapping(
        workspace=workspace, compiled_lock_sha256=compiled_lock_sha256, env=env,
        calibration_digest=calibration_digest,
    )
    for key, value in current.items():
        if receipt.get(key) != value:
            label = key.replace("_", " ")
            return (
                f"{label} no longer matches the oracle receipt "
                f"(got {value!r}, receipt had {receipt.get(key)!r})"
            )
    return None


def _default_confirm(prompt: str) -> bool:
    """Default TTY confirmation: prompt on stdin, truthy answer confirms."""
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _default_docker_ok() -> bool:
    """Default Docker allowlist probe: never fall back to public networking.

    The benchmark runtime is docker-allowlisted by platform contract; this is
    the injectable default used when the orchestrator did not supply one.
    """
    return True


def _default_spawn(cmd: list[str], *, cwd: Path, env: dict[str, Any]) -> dict[str, Any]:
    """Default Harbor subprocess spawn (the real production seam)."""
    completed = subprocess.run(cmd, cwd=str(cwd), env=dict(env))
    return {"returncode": completed.returncode}


def _calibration_digest(workspace: Path) -> str:
    """sha256 of ``runtime/calibration-receipt.json`` bytes (empty when absent)."""
    path = workspace / "runtime" / "calibration-receipt.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _ledger_job_dir(workspace: Path, run_id: str) -> Path:
    """The job dir recorded for ``run_id`` (the ledger, not an mtime guess).

    Selecting by the ledger rather than the newest-mtime ``jobs/`` dir means a
    spawn that wrote nothing cannot cause a prior run's job dir to be attested
    by this run's oracle receipt.
    """
    doc = _load_ledger(workspace)
    for run in doc["runs"]:
        if run.get("run_id") == run_id:
            return Path(run["job_dir"])
    raise RunError(f"run {run_id!r} not found in cleanup ledger {_ledger_path(workspace)}")


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
    """Supervised Harbor run: fail-closed preflight, then one gated run.

    The Oracle path (``oracle=True``) runs a self-match pass to prove the stack
    reproduces gold and writes ``harbor/oracle-receipt.json`` on success;
    the default path is gated by a matching Oracle receipt before any paid call.
    Every run is recorded in ``runtime/harbor.json`` (a unique contained job
    dir), and Harbor's exit code is preserved on failure.
    """
    workspace = Path(workspace).resolve()
    env = dict(env) if env is not None else {}
    docker_ok = docker_ok or _default_docker_ok
    confirm = confirm or _default_confirm
    spawn = spawn or _default_spawn

    # 1. Fail-closed preflight (no running entry, no spawn on failure).
    failures = _preflight(workspace, oracle=oracle, env=env, docker_ok=docker_ok)
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    # 2. Pre-run spend summary.
    print(_pre_run_summary(workspace, env=env))

    # 3. Confirmation gate (still before any paid call).
    if not yes:
        mode = "the Oracle self-match" if oracle else "the paid benchmark"
        if not confirm(f"Refusing unconfirmed {mode} Harbor run"):
            print("run cancelled by user", file=sys.stderr)
            return 1

    # 4. Default (non-Oracle) runs gate on a prior Oracle receipt first.
    compiled_lock_sha = _compiled_lock_sha256(workspace)
    if not oracle:
        gate_reason = _default_run_gate(
            workspace, env=env, compiled_lock_sha256=compiled_lock_sha,
            calibration_digest=_calibration_digest(workspace),
        )
        if gate_reason is not None:
            print(gate_reason, file=sys.stderr)
            return 1

    # 5. Assign a unique contained job dir and append the running ledger row.
    run_id = str(uuid.uuid4())
    job_dir = (workspace / "harbor" / "jobs" / run_id).resolve()
    mode = "oracle" if oracle else "benchmark"
    ledger_append_running(workspace, run_id=run_id, compiled_lock_sha256=compiled_lock_sha,
                          job_dir=str(job_dir), mode=mode,
                          profile_digest=env.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"),
                          reviewer_effort=env.get("DAYDREAM_REVIEW_EFFORT"))

    # 6. Spawn Harbor with an absolute config path, the parent environment
    #    (PATH/HOME/etc.) merged in, and telemetry forced off.
    config = (workspace / "harbor" / ("harbor-oracle.yaml" if oracle else "harbor-job.yaml")).resolve()
    harbor_exe = package.resolve_harbor()
    spawn_env = {key: value for key, value in os.environ.items()}
    for key, value in env.items():
        if value is not None:
            spawn_env[key] = value
    spawn_env["HARBOR_TELEMETRY"] = "off"
    result = spawn(
        [harbor_exe, "run", "-c", str(config)],
        cwd=workspace / "harbor", env=spawn_env,
    )
    returncode = int(result.get("returncode", 0))

    # 7. Post-run: parse results and reconcile the ledger / receipt. The
    #    exact resolved trial environments are persisted into the ledger on
    #    every terminal mark so ``clean --jobs`` has recorded image refs.
    try:
        actual_dir = _ledger_job_dir(workspace, run_id)
        if oracle:
            ok, environments = _parse_job_results(actual_dir)
            if returncode != 0 or not ok:
                if returncode == 0:
                    print(
                        "Oracle run did not reproduce gold; no oracle-receipt.json "
                        "was written.",
                        file=sys.stderr,
                    )
                ledger_mark(workspace, run_id, state="cleanup_pending",
                            environments=environments)
                return returncode or 1
            write_code = _write_oracle_receipt(
                workspace, job_dir=actual_dir, compiled_lock_sha256=compiled_lock_sha,
                env=env, calibration_digest=_calibration_digest(workspace),
            )
            ledger_mark(workspace, run_id, state="complete",
                        environments=environments)
            return write_code or returncode
        # Default real run: preserve Harbor's exact exit code. Failing and
        # successful runs persist the same resolved trial environments;
        # ``_parse_job_results`` already returns ``(False, [])`` for a missing
        # job dir, so no is_dir() pre-check is needed.
        _, environments = _parse_job_results(actual_dir)
        ledger_mark(
            workspace, run_id,
            state="cleanup_pending" if returncode != 0 else "complete",
            environments=environments,
        )
        return returncode
    except Exception:
        print("unexpected error during supervised run", file=sys.stderr)
        try:
            ledger_mark(workspace, run_id, state="cleanup_pending", error="unexpected error")
        except RunError:
            pass
        raise

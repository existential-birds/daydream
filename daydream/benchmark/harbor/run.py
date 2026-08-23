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
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Callable

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


def _compiled_lock_sha256(workspace: Path) -> str:
    """sha256 of the compiled ``harbor/benchmark.lock.json`` bytes."""
    return hashlib.sha256((workspace / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()


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
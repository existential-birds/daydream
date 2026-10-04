"""Load repository policy from pyproject's [tool.daydream] and .daydream.toml.

The dotfile wins per key. Phase tables merge individually; improve and diagram
also merge per key, preserving lower-precedence siblings.
"""

from __future__ import annotations

import logging
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.config import DEFAULT_RETRY_RECOVERY_ALLOWANCE_S
from daydream.retry_policy import (
    decode_retry_recovery_allowance,
    undeclared_retry_allowance_message,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaydreamFileConfig:
    """Merged repository policy; None means the runtime default still applies.

    CLI globals outrank file phase overrides, which outrank file globals.
    ``phases`` also accepts extension phase keys. Operator destinations and
    credentials (tracing, Hub uploads) never come from repository settings.

    Scalar policy is lenient: malformed values degrade to unset. Boolean
    switches require actual booleans, never truthy numbers. Counts reject
    booleans; budgets and quality thresholds also reject NaN and infinity. The
    repair-job limits follow exactly those two policies:
    ``repair_execution_wall_s``/``repair_job_wall_s`` are budgets,
    ``repair_max_executions`` is a count, and ``repair_grant_job_wall_s`` is an
    operator's additional finite allowance for a job that ran out of time.
    Non-negative bounds preserve zero: a zero group budget intentionally skips
    fixes. Test-command timeouts and diagram/improve bounds must be positive.
    Invalid declared retry allowances warn before using the default.

    ``latency_profile`` is deliberately parsed without validating its vocabulary:
    the runtime resolver promotes unknown names to forensic. ``review_profile``
    is only a path here; review_profile.py owns strict profile validation.
    ``extra_risk_categories`` likewise gets fail-loud vocabulary validation at
    verification, and can only widen mandatory selection.

    Improve service roots/groups and partition bounds come from ``[improve]``;
    issue publication requires explicit ``[improve.github] publish_issues=true``.
    ``[diagram] mode`` allows auto/off; forcing a diagram kind is CLI-only.
    Empty diagram service roots inherit Improve roots, then layout inference.

    ``test_required_suites`` declares which suites the single configured command
    covers; it never creates a second runner or substitutes a targeted check.
    An explicit false review-cache switch wins at either CLI or file tier, while
    cache retention bounds remain independent of the enable switch.
    Field defaults and the full configuration reference live in config.py and README.
    """

    model: str | None = None
    backend: str | None = None
    reasoning_effort: str | None = None
    latency_profile: str | None = None
    phases: dict[str, dict[str, str]] = field(default_factory=dict)
    shallow_fanout_threshold: int | None = None
    precision_mode: bool | None = None
    approve_on_clean: bool | None = None
    scope_issue_filing: bool | None = None
    group_max_wall_s: float | None = None
    group_max_serial_items: int | None = None
    repair_execution_wall_s: float | None = None
    repair_job_wall_s: float | None = None
    repair_max_executions: int | None = None
    repair_grant_job_wall_s: float | None = None
    retry_recovery_allowance_s: float | None = None
    deep_shard_enabled: bool | None = None
    deep_shard_max_files: int | None = None
    deep_shard_max_bytes: int | None = None
    deep_shard_fanout_cap: int | None = None
    deep_shard_frontier_max: int | None = None
    review_cache_enabled: bool | None = None
    review_cache_max_entries: int | None = None
    review_cache_max_bytes: int | None = None
    review_cache_max_age_days: int | None = None
    verify_all: bool | None = None
    extra_risk_categories: list[str] = field(default_factory=list)
    quality_gate_enabled: bool | None = None
    quality_gate_erosion_delta: float | None = None
    quality_gate_verbosity_delta: float | None = None
    quality_gate_erosion_absolute: float | None = None
    quality_gate_verbosity_absolute: float | None = None
    supervisor: str | None = None
    supervisor_deny_globs: list[str] = field(default_factory=list)
    tool_supervisor: str | None = None
    tool_bash_deny: list[str] = field(default_factory=list)
    improve_service_roots: list[str] = field(default_factory=list)
    improve_service_groups: dict[str, list[str]] = field(default_factory=dict)
    improve_partition_max_files: int | None = None
    improve_max_partition_groups: int | None = None
    improve_github_publish_issues: bool = False
    review_profile: Path | None = None
    diagram_mode: str | None = None
    diagram_min_code_files: int | None = None
    diagram_min_modules: int | None = None
    diagram_min_branch_points: int | None = None
    diagram_service_roots: list[str] = field(default_factory=list)
    test_command: str | None = None
    test_command_wall_s: float | None = None
    test_required_suites: list[str] = field(default_factory=list)

    def phase_model(self, phase: str) -> str | None:
        """Return the configured model for a phase."""
        return self.phases.get(phase, {}).get("model")

    def phase_backend(self, phase: str) -> str | None:
        """Return the configured backend for a phase."""
        return self.phases.get(phase, {}).get("backend")

    def phase_reasoning_effort(self, phase: str) -> str | None:
        """Return the configured reasoning effort for a phase."""
        return self.phases.get(phase, {}).get("reasoning_effort")


def load_toml_or_empty(path: Path) -> dict[str, Any]:
    """Read optional TOML without failing callers such as pricing and workspace-copy config.

    Absent files return {}; malformed or unreadable files warn and return {}.
    Use _load_toml when malformed configuration must raise.
    """
    try:
        return _load_toml(path)
    except ValueError as exc:
        logger.warning("daydream: malformed TOML in %s — ignoring (%s)", path, exc)
        return {}
    except OSError as exc:
        logger.warning("daydream: could not read %s — ignoring (%s)", path, exc)
        return {}


def _load_toml(path: Path) -> dict[str, Any]:
    """Read TOML or return {} when absent; malformed files raise ValueError naming the path."""
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Malformed TOML in {path.name}: {exc}") from exc


def _merge_section(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge scalar keys and improve/diagram tables, then merge phases individually."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key == "phases" and isinstance(value, dict) and isinstance(merged.get("phases"), dict):
            phases: dict[str, Any] = dict(merged["phases"])
            for phase_name, phase_table in value.items():
                if isinstance(phase_table, dict) and isinstance(phases.get(phase_name), dict):
                    phases[phase_name] = {**phases[phase_name], **phase_table}
                else:
                    phases[phase_name] = phase_table
            merged["phases"] = phases
        elif key in ("improve", "diagram") and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _coerce_phases(raw: Any) -> dict[str, dict[str, str]]:
    """Normalize a raw ``phases`` value into ``dict[str, dict[str, str]]``."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for phase_name, table in raw.items():
        if isinstance(table, dict):
            result[str(phase_name)] = {str(k): str(v) for k, v in table.items()}
        else:
            logger.warning(
                "daydream config: ignoring phase %r — expected a table, got %s",
                phase_name,
                type(table).__name__,
            )
    return result


def _coerce_int(raw: Any) -> int | None:
    """Return ``raw`` as an int, or None for bool/non-int (degrade to default)."""
    if isinstance(raw, bool):
        return None
    return raw if isinstance(raw, int) else None


def _coerce_non_negative_int(raw: Any) -> int | None:
    """Preserve integer zero; reject negative values, booleans, and non-integers."""
    if isinstance(raw, bool):
        return None
    return raw if isinstance(raw, int) and raw >= 0 else None


def _coerce_optional_bool(raw: Any) -> bool | None:
    """Accept only actual booleans: precision_mode=1 must remain unset."""
    return raw if isinstance(raw, bool) else None


def _coerce_positive_int(table: dict[str, Any], key: str) -> int | None:
    """Read a positive integer, with hyphenated spelling as fallback."""
    value = _coerce_int(table.get(key, table.get(key.replace("_", "-"))))
    return value if value is not None and value > 0 else None


def _coerce_float(raw: Any) -> float | None:
    """Accept integer/float budgets, rejecting booleans and non-numbers."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _coerce_non_negative_float(raw: Any) -> float | None:
    """Accept finite non-negative budgets, including zero to deliberately skip fixes."""
    value = _coerce_float(raw)
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return value


def _coerce_retry_recovery_allowance(merged: dict[str, Any]) -> float | None:
    """Use the shared retry decoder; invalid declared values warn, absent keys stay silent."""
    raw = merged.get("retry_recovery_allowance_s")
    value = None if raw is None else decode_retry_recovery_allowance(raw)
    if raw is not None and value is None:
        logger.warning(
            "daydream config: %s; using default %s",
            undeclared_retry_allowance_message("retry_recovery_allowance_s", raw),
            DEFAULT_RETRY_RECOVERY_ALLOWANCE_S,
        )
    return value


def _coerce_positive_float(raw: Any) -> float | None:
    """Reject non-finite or non-positive timeouts: they disable or immediately expire the deadline."""
    value = _coerce_float(raw)
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return value


def _coerce_string(raw: Any) -> str | None:
    """Return a non-empty string, or None for absent/malformed values."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw


def _coerce_string_list(raw: Any) -> list[str]:
    """Return a list of strings, or an empty list for malformed values."""
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        return []
    return list(raw)


def _coerce_choice(raw: Any, choices: set[str]) -> str | None:
    """Return a configured choice, or None for an invalid value."""
    return raw if isinstance(raw, str) and raw in choices else None


def _coerce_review_profile_path(raw: Any) -> Path | None:
    """Read a nonblank profile path; strict profile validation belongs to review_profile.py."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw)


def load_file_config(root: Path) -> DaydreamFileConfig:
    """Merge repository policy, returning empty defaults when neither file exists.

    Malformed TOML raises ValueError naming the offending file.
    """
    pyproject = _load_toml(root / "pyproject.toml")
    tool_section = pyproject.get("tool", {})
    base = tool_section.get("daydream", {}) if isinstance(tool_section, dict) else {}
    base = base if isinstance(base, dict) else {}

    dotfile = _load_toml(root / ".daydream.toml")

    merged = _merge_section(base, dotfile)

    # Warn on removed policy rather than silently accepting a stale configuration.
    if "bench" in merged:
        logger.warning(
            "daydream: [tool.daydream.bench] is no longer a supported daydream "
            "config section (legacy benchmark verb removed); ignoring it"
        )

    model = merged.get("model")
    backend = merged.get("backend")
    reasoning_effort = merged.get("reasoning_effort")
    threshold = _coerce_int(merged.get("shallow_fanout_threshold"))
    # Non-table sections behave as absent, preserving defaults.
    diagram = merged.get("diagram")
    diagram = diagram if isinstance(diagram, dict) else {}
    improve = merged.get("improve")
    improve = improve if isinstance(improve, dict) else {}
    service_groups = improve.get("service_groups")
    service_groups = service_groups if isinstance(service_groups, dict) else {}
    improve_github = improve.get("github")
    improve_github = improve_github if isinstance(improve_github, dict) else {}
    raw_improve_publish = improve_github.get(
        "publish_issues", improve_github.get("publish-issues")
    )
    improve_github_publish_issues = (
        raw_improve_publish if isinstance(raw_improve_publish, bool) else False
    )
    return DaydreamFileConfig(
        model=str(model) if model is not None else None,
        backend=str(backend) if backend is not None else None,
        reasoning_effort=str(reasoning_effort) if reasoning_effort is not None else None,
        latency_profile=_coerce_string(merged.get("latency_profile")),
        phases=_coerce_phases(merged.get("phases")),
        shallow_fanout_threshold=threshold,
        precision_mode=_coerce_optional_bool(merged.get("precision_mode")),
        approve_on_clean=_coerce_optional_bool(merged.get("approve_on_clean")),
        scope_issue_filing=_coerce_optional_bool(merged.get("scope_issue_filing")),
        group_max_wall_s=_coerce_non_negative_float(merged.get("group_max_wall_s")),
        group_max_serial_items=_coerce_non_negative_int(merged.get("group_max_serial_items")),
        repair_execution_wall_s=_coerce_non_negative_float(merged.get("repair_execution_wall_s")),
        repair_job_wall_s=_coerce_non_negative_float(merged.get("repair_job_wall_s")),
        repair_max_executions=_coerce_non_negative_int(merged.get("repair_max_executions")),
        repair_grant_job_wall_s=_coerce_non_negative_float(merged.get("repair_grant_job_wall_s")),
        retry_recovery_allowance_s=_coerce_retry_recovery_allowance(merged),
        review_profile=_coerce_review_profile_path(merged.get("review_profile")),
        deep_shard_enabled=_coerce_optional_bool(merged.get("deep_shard_enabled")),
        deep_shard_max_files=_coerce_non_negative_int(merged.get("deep_shard_max_files")),
        deep_shard_max_bytes=_coerce_non_negative_int(merged.get("deep_shard_max_bytes")),
        deep_shard_fanout_cap=_coerce_non_negative_int(merged.get("deep_shard_fanout_cap")),
        deep_shard_frontier_max=_coerce_non_negative_int(merged.get("deep_shard_frontier_max")),
        review_cache_enabled=_coerce_optional_bool(merged.get("review_cache_enabled")),
        review_cache_max_entries=_coerce_non_negative_int(merged.get("review_cache_max_entries")),
        review_cache_max_bytes=_coerce_non_negative_int(merged.get("review_cache_max_bytes")),
        review_cache_max_age_days=_coerce_non_negative_int(merged.get("review_cache_max_age_days")),
        verify_all=_coerce_optional_bool(merged.get("verify_all")),
        extra_risk_categories=_coerce_string_list(merged.get("extra_risk_categories")),
        quality_gate_enabled=_coerce_optional_bool(merged.get("quality_gate_enabled")),
        quality_gate_erosion_delta=_coerce_non_negative_float(merged.get("quality_gate_erosion_delta")),
        quality_gate_verbosity_delta=_coerce_non_negative_float(merged.get("quality_gate_verbosity_delta")),
        quality_gate_erosion_absolute=_coerce_non_negative_float(merged.get("quality_gate_erosion_absolute")),
        quality_gate_verbosity_absolute=_coerce_non_negative_float(merged.get("quality_gate_verbosity_absolute")),
        supervisor=_coerce_choice(merged.get("supervisor"), {"off", "rules", "llm"}),
        supervisor_deny_globs=_coerce_string_list(merged.get("supervisor_deny_globs")),
        tool_supervisor=_coerce_choice(merged.get("tool_supervisor"), {"off", "rules"}),
        tool_bash_deny=_coerce_string_list(merged.get("tool_bash_deny")),
        improve_service_roots=_coerce_string_list(improve.get("service_roots")),
        improve_service_groups={
            str(group): _coerce_string_list(roots)
            for group, roots in service_groups.items()
        },
        improve_partition_max_files=_coerce_positive_int(improve, "partition_max_files"),
        improve_max_partition_groups=_coerce_positive_int(improve, "max_partition_groups"),
        improve_github_publish_issues=improve_github_publish_issues,
        diagram_mode=_coerce_choice(diagram.get("mode"), {"auto", "off"}),
        diagram_min_code_files=_coerce_positive_int(diagram, "min_code_files"),
        diagram_min_modules=_coerce_positive_int(diagram, "min_modules"),
        diagram_min_branch_points=_coerce_positive_int(diagram, "min_branch_points"),
        diagram_service_roots=_coerce_string_list(diagram.get("service_roots")),
        test_command=_coerce_string(merged.get("test_command")),
        test_command_wall_s=_coerce_positive_float(merged.get("test_command_wall_s")),
        test_required_suites=_coerce_string_list(merged.get("test_required_suites")),
    )

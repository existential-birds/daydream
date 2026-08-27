"""Config-file loader for daydream.

Loads daydream settings from two on-disk sources, merged per-key:

1. ``pyproject.toml`` under the ``[tool.daydream]`` table (low precedence).
2. ``.daydream.toml`` at the repo root (root keys; high precedence).

The dotfile's keys override pyproject's, merged **per-key** so a
``[phases.fix]`` table in the dotfile does not wipe a ``[phases.review]``
table declared in pyproject.

Exports:
    DaydreamFileConfig: frozen config snapshot with phase accessors.
    load_file_config: read + merge both sources from a repo root.
"""

from __future__ import annotations

import logging
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaydreamFileConfig:
    """Immutable snapshot of file-sourced daydream configuration.

    Attributes:
        model: Global default model, or None if unset.
        backend: Global default backend name, or None if unset.
        reasoning_effort: Global default reasoning-effort override, or None if
            unset. Only consumed by the Codex backend today (forwarded as
            ``-c model_reasoning_effort=...``); ignored for claude/pi.
        phases: Per-phase sub-tables mapping phase name to a dict of keys
            (e.g. ``{"fix": {"backend": "codex", "model": "..."}}``).
        shallow_fanout_threshold: Max changed-file count that triggers the
            tiny-diff short-circuit in deep mode (issue #172). ``None`` falls
            through to the RunConfig field / orchestrator default. ``0``
            explicitly disables the short-circuit.
        precision_mode: Opt-in precision suppression (issue #232). ``None`` falls
            through to the RunConfig field / orchestrator default; ``True`` runs
            the skeptical suppression pass over borderline findings.
        approve_on_clean: Opt-in approval of clean deep reviews (issue #343).
            ``None`` falls through to the RunConfig field / orchestrator default;
            ``True`` posts ``event: "APPROVE"`` when a deep review has zero
            high/medium findings. Explicit opt-in: never coerced from a non-bool.
        review_profile: Repo-committed review-profile path (R9). A lenient path
            read only — the strict profile parse + validation stays in
            ``review_profile.py``. ``None`` (absent or non-str) means the key is
            unset and the default profile applies.
        group_max_wall_s: Per-file-group fix wall-clock ceiling in seconds
            (issue #201), a global ``[tool.daydream]`` key. Bounds the cumulative
            wall-clock of all fix ``run_agent`` turns targeting one file group so
            a runaway file cannot dominate a run. ``None`` (the default when the
            key is absent or junk) falls through to
            ``config.DEFAULT_GROUP_MAX_WALL_S`` (600.0).
        group_max_serial_items: Per-file-group serial fix-call ceiling (#201), a
            global ``[tool.daydream]`` key. Caps the number of per-finding fix
            calls in one file group (the group is severity-sorted, so the dropped
            tail is lowest-severity). ``None`` (the default when the key is absent
            or junk) falls through to ``config.DEFAULT_GROUP_MAX_SERIAL_ITEMS`` (6).
        uncovered_sweep: Issue #309. Toggle the uncovered-diff-file sweep
            (second-pass reviewer over diff files no per-stack reviewer read).
            ``None`` falls through to the RunConfig field / orchestrator default
            (``True``); ``False`` disables the pass.
        uncovered_sweep_max_files: Issue #309. Cap on how many uncovered files
            are swept in one run. Non-negative only: ``0`` disables the sweep;
            a negative value degrades to ``None`` (the named default applies).
            ``None`` falls through to the RunConfig field /
            ``config.DEFAULT_UNCOVERED_SWEEP_MAX_FILES`` (10).
        uncovered_sweep_min_hunk_lines: Issue #309. Minimum added/removed lines
            a file's hunks must contain to warrant a sweep. Non-negative only:
            ``0`` removes the floor; a negative value degrades to ``None`` (the
            named default applies). ``None`` falls through to the RunConfig field /
            ``config.DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES`` (5).
        quality_gate_enabled: Issue #315. Toggle the fix-phase anti-degradation
            quality gate. ``None`` falls through to the orchestrator default
            (``config.DEFAULT_QUALITY_GATE_ENABLED``, ``True``); ``False`` skips
            the whole computation and writes ``{"enabled": false}``.
        quality_gate_erosion_delta: Issue #315. Per-file erosion-delta threshold
            above which a fixed file is flagged. Finite non-negative only: a
            negative, NaN, or infinite value degrades to ``None`` (the named
            default applies) so a bad threshold can neither flag every unchanged
            file (a negative floor is exceeded by any delta) nor silently
            disable the metric (every comparison against NaN is False).
            ``None`` falls through to
            ``config.DEFAULT_QUALITY_GATE_EROSION_DELTA`` (0.05).
        quality_gate_verbosity_delta: Issue #315. Per-file verbosity-delta
            threshold above which a fixed file is flagged. Finite non-negative
            only, same degrade-to-``None`` rule as ``quality_gate_erosion_delta``.
            ``None`` falls through to
            ``config.DEFAULT_QUALITY_GATE_VERBOSITY_DELTA`` (0.05).
        quality_gate_erosion_absolute: Issue #315. Absolute post-fix erosion
            threshold for the undefined-baseline fallback -- the BEFORE erosion
            metric is ``None`` (no functions pre-fix), so no delta exists and
            the AFTER value is compared against this knob instead of the delta
            one (#329 / CodeRabbit Finding D). Finite non-negative only, same
            degrade-to-``None`` rule as ``quality_gate_erosion_delta``. ``None``
            falls through to
            ``config.DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE`` (0.05).
        quality_gate_verbosity_absolute: Issue #315. Absolute post-fix verbosity
            threshold for the undefined-baseline fallback, the verbosity twin of
            ``quality_gate_erosion_absolute``. Finite non-negative only, same
            degrade-to-``None`` rule. ``None`` falls through to
            ``config.DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE`` (0.05).
        supervisor: Findings supervisor mode (``"off"``, ``"rules"``, or
            ``"llm"``), or None when unset/invalid.
        supervisor_deny_globs: Repository-relative deny globs shared by findings
            and tool supervision.
        tool_supervisor: Built-in tool supervisor mode (``"off"`` or ``"rules"``),
            or None when unset/invalid.
        tool_bash_deny: Regular expressions for denied Bash commands.
        improve_service_roots: Repository-relative glob patterns identifying
            service roots for the improve flow.
        improve_service_groups: Named groups of repository-relative service roots.
        improve_github_publish_issues: Whether Improve should publish each validated
            local plan as a GitHub issue. This is an explicit repository-level
            opt-in under ``[tool.daydream.improve.github]``; absent or malformed
            values leave publishing disabled.
    """

    model: str | None = None
    backend: str | None = None
    reasoning_effort: str | None = None
    phases: dict[str, dict[str, str]] = field(default_factory=dict)
    shallow_fanout_threshold: int | None = None
    precision_mode: bool | None = None
    approve_on_clean: bool | None = None
    group_max_wall_s: float | None = None
    group_max_serial_items: int | None = None
    uncovered_sweep: bool | None = None
    uncovered_sweep_max_files: int | None = None
    uncovered_sweep_min_hunk_lines: int | None = None
    deep_shard_enabled: bool | None = None
    deep_shard_max_files: int | None = None
    deep_shard_max_bytes: int | None = None
    deep_shard_fanout_cap: int | None = None
    deep_shard_frontier_max: int | None = None
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
    """Parse a TOML file into a dict, returning {} when absent or malformed.

    Never raises: callers that must not break on a bad user file (e.g. price
    overrides, workspace copy config) use this instead of the error-raising
    :func:`_load_toml`.

    Raises:
        Never. Malformed TOML is logged as a warning and yields ``{}``.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        logger.warning("daydream: malformed TOML in %s — ignoring (%s)", path, exc)
        return {}
    except OSError as exc:
        logger.warning("daydream: could not read %s — ignoring (%s)", path, exc)
        return {}


def _load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file into a dict, returning ``{}`` when the file is absent.

    Raises:
        ValueError: If the file exists but is malformed; the message names
            the offending file.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Malformed TOML in {path.name}: {exc}") from exc


def _merge_section(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two daydream config sections, override winning per-key.

    Scalar keys (``model``, ``backend``) from ``override`` replace ``base``.
    The ``phases`` sub-table is merged per-phase so an override phase table
    does not discard phases declared only in ``base``. The ``improve``
    sub-table is likewise merged per-key.
    """
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
        elif key == "improve" and isinstance(value, dict) and isinstance(merged.get("improve"), dict):
            merged["improve"] = {**merged["improve"], **value}
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
    """Return ``raw`` as a non-negative int, or None otherwise (degrade to default).

    Unlike :func:`_coerce_int`, a negative value degrades to ``None`` so the
    named default applies (issue #309): a negative sweep capacity cap or hunk
    floor is never a meaningful count. Explicit ``0`` is preserved — ``0`` max
    files means "sweep nothing" and ``0`` min hunk lines means "no hunk-size
    floor".
    """
    if isinstance(raw, bool):
        return None
    return raw if isinstance(raw, int) and raw >= 0 else None


def _coerce_positive_int(table: dict[str, Any], key: str) -> int | None:
    """Return a positive int bound from ``key``, accepting its hyphenated spelling.

    Non-int and non-positive values degrade to None (the built-in default then
    applies), matching the loader's lenient coercion style.
    """
    value = _coerce_int(table.get(key, table.get(key.replace("_", "-"))))
    return value if value is not None and value > 0 else None


def _coerce_float(raw: Any) -> float | None:
    """Return ``raw`` as a float, or None for bool/non-number (degrade to default).

    Accepts TOML ints and floats (an int budget like ``group_max_wall_s = 600``
    round-trips to ``600.0``); rejects bool and everything else.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _coerce_quality_threshold(raw: Any) -> float | None:
    """Return ``raw`` as a finite non-negative float, else None (degrade to default).

    Quality-gate thresholds must be finite and non-negative
    (#329 / Finding 7): a NaN or infinite threshold makes every ``>``
    comparison False, silently disabling the metric (and non-standard ``NaN``
    reaches JSON); a negative threshold makes a zero delta (an unchanged file)
    exceed it, flagging files that did not regress. Anything invalid --
    negative, NaN, inf, bool, or non-number -- degrades to ``None`` so the
    ``config.py`` default applies.
    """
    value = _coerce_float(raw)
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return value


def _coerce_string_list(raw: Any) -> list[str]:
    """Return a list of strings, or an empty list for malformed values."""
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        return []
    return list(raw)


def _coerce_choice(raw: Any, choices: set[str]) -> str | None:
    """Return a configured choice, or None for an invalid value."""
    return raw if isinstance(raw, str) and raw in choices else None


def _coerce_review_profile_path(raw: Any) -> Path | None:
    """Leniently read a repo-committed review-profile path (R9).

    A path string only — the strict profile parse stays in review_profile.py.
    Malformed/non-string values degrade to None (unset).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw)


def load_file_config(root: Path) -> DaydreamFileConfig:
    """Load and merge daydream file configuration from a repo root.

    Reads ``root/pyproject.toml`` ``[tool.daydream]`` (low precedence) and
    ``root/.daydream.toml`` (high precedence), merging the two per-key. The
    dotfile wins on conflicting scalar keys; phase sub-tables merge so each
    source contributes its own phases. Absent files yield an empty config.

    Raises:
        ValueError: If a present config file is malformed TOML; the message
            names the offending file.
    """
    pyproject = _load_toml(root / "pyproject.toml")
    tool_section = pyproject.get("tool", {})
    base = tool_section.get("daydream", {}) if isinstance(tool_section, dict) else {}
    base = base if isinstance(base, dict) else {}

    dotfile = _load_toml(root / ".daydream.toml")

    merged = _merge_section(base, dotfile)

    # The legacy `bench` table was removed with the Martian benchmark stack
    # (issue-785). A stale `[tool.daydream.bench]` section is now ignored; warn
    # so the upgrade path is not silent, mirroring cli.py's loud rejection of the
    # removed legacy `bench` verb.
    if "bench" in merged:
        logger.warning(
            "daydream: [tool.daydream.bench] is no longer a supported daydream "
            "config section (legacy benchmark verb removed); ignoring it"
        )

    model = merged.get("model")
    backend = merged.get("backend")
    reasoning_effort = merged.get("reasoning_effort")
    threshold = _coerce_int(merged.get("shallow_fanout_threshold"))
    # precision_mode: bool only. Any non-bool value (str, int, list) degrades to
    # None rather than crashing the loader; truthy ints are *not* coerced to True
    # so an accidental ``precision_mode = 1`` is treated as unset, not enabled.
    raw_precision = merged.get("precision_mode")
    precision: bool | None = raw_precision if isinstance(raw_precision, bool) else None
    # approve_on_clean: bool only, same degrade-to-None rule as precision_mode so
    # an accidental ``approve_on_clean = 1`` is treated as unset, not enabled.
    raw_approve = merged.get("approve_on_clean")
    approve_on_clean: bool | None = raw_approve if isinstance(raw_approve, bool) else None
    # uncovered_sweep: bool only, same degrade-to-None rule as precision_mode so
    # an accidental ``uncovered_sweep = 1`` is treated as unset, not enabled.
    raw_uncovered_sweep = merged.get("uncovered_sweep")
    uncovered_sweep: bool | None = (
        raw_uncovered_sweep if isinstance(raw_uncovered_sweep, bool) else None
    )
    # deep_shard_enabled: bool only, same degrade-to-None rule as precision_mode
    # so an accidental ``deep_shard_enabled = 1`` is treated as unset, not
    # enabled. The shard bounds are non-negative ints (reject bool/float via
    # _coerce_non_negative_int, mirroring uncovered_sweep_max_files).
    raw_deep_shard_enabled = merged.get("deep_shard_enabled")
    deep_shard_enabled: bool | None = (
        raw_deep_shard_enabled if isinstance(raw_deep_shard_enabled, bool) else None
    )
    # quality_gate_enabled: bool only, same degrade-to-None rule. The delta AND
    # absolute thresholds are finite non-negative floats (reject bool, coerce
    # ints, reject negative / NaN / inf) via _coerce_quality_threshold.
    raw_quality_gate_enabled = merged.get("quality_gate_enabled")
    quality_gate_enabled: bool | None = (
        raw_quality_gate_enabled if isinstance(raw_quality_gate_enabled, bool) else None
    )
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
    # Per-file-group fix budgets (#201): tolerate junk by degrading to None (the
    # config.py default then applies). bool is excluded even though it subclasses
    # int/float — ``group_max_serial_items = true`` is never a meaningful count.
    return DaydreamFileConfig(
        model=str(model) if model is not None else None,
        backend=str(backend) if backend is not None else None,
        reasoning_effort=str(reasoning_effort) if reasoning_effort is not None else None,
        phases=_coerce_phases(merged.get("phases")),
        shallow_fanout_threshold=threshold,
        precision_mode=precision,
        approve_on_clean=approve_on_clean,
        group_max_wall_s=_coerce_float(merged.get("group_max_wall_s")),
        group_max_serial_items=_coerce_int(merged.get("group_max_serial_items")),
        review_profile=_coerce_review_profile_path(merged.get("review_profile")),
        uncovered_sweep=uncovered_sweep,
        uncovered_sweep_max_files=_coerce_non_negative_int(merged.get("uncovered_sweep_max_files")),
        uncovered_sweep_min_hunk_lines=_coerce_non_negative_int(merged.get("uncovered_sweep_min_hunk_lines")),
        deep_shard_enabled=deep_shard_enabled,
        deep_shard_max_files=_coerce_non_negative_int(merged.get("deep_shard_max_files")),
        deep_shard_max_bytes=_coerce_non_negative_int(merged.get("deep_shard_max_bytes")),
        deep_shard_fanout_cap=_coerce_non_negative_int(merged.get("deep_shard_fanout_cap")),
        deep_shard_frontier_max=_coerce_non_negative_int(merged.get("deep_shard_frontier_max")),
        quality_gate_enabled=quality_gate_enabled,
        quality_gate_erosion_delta=_coerce_quality_threshold(merged.get("quality_gate_erosion_delta")),
        quality_gate_verbosity_delta=_coerce_quality_threshold(merged.get("quality_gate_verbosity_delta")),
        quality_gate_erosion_absolute=_coerce_quality_threshold(merged.get("quality_gate_erosion_absolute")),
        quality_gate_verbosity_absolute=_coerce_quality_threshold(merged.get("quality_gate_verbosity_absolute")),
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
    )

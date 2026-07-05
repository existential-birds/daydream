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
        phases: Per-phase sub-tables mapping phase name to a dict of keys
            (e.g. ``{"fix": {"backend": "codex", "model": "..."}}``).
        shallow_fanout_threshold: Max changed-file count that triggers the
            tiny-diff short-circuit in deep mode (issue #172). ``None`` falls
            through to the RunConfig field / orchestrator default. ``0``
            explicitly disables the short-circuit.
        precision_mode: Opt-in precision suppression (issue #232). ``None`` falls
            through to the RunConfig field / orchestrator default; ``True`` runs
            the skeptical suppression pass over borderline findings.
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
    """

    model: str | None = None
    backend: str | None = None
    phases: dict[str, dict[str, str]] = field(default_factory=dict)
    bench: dict[str, Any] = field(default_factory=dict)
    shallow_fanout_threshold: int | None = None
    precision_mode: bool | None = None
    group_max_wall_s: float | None = None
    group_max_serial_items: int | None = None

    def phase_model(self, phase: str) -> str | None:
        """Return the configured model for a phase.

        Args:
            phase: Phase name to look up (e.g. ``"fix"``).

        Returns:
            The configured model, or None if the phase has no ``model`` key.
        """
        return self.phases.get(phase, {}).get("model")

    def phase_backend(self, phase: str) -> str | None:
        """Return the configured backend for a phase.

        Args:
            phase: Phase name to look up (e.g. ``"fix"``).

        Returns:
            The configured backend, or None if the phase has no ``backend`` key.
        """
        return self.phases.get(phase, {}).get("backend")


def load_toml_or_empty(path: Path) -> dict[str, Any]:
    """Parse a TOML file into a dict, returning {} when absent or malformed.

    Never raises: callers that must not break on a bad user file (e.g. price
    overrides, workspace copy config) use this instead of the error-raising
    :func:`_load_toml`.

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed table, or an empty dict if the file is absent or malformed.

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
    """Parse a TOML file into a dict.

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed table, or an empty dict if the file is absent.

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
    does not discard phases declared only in ``base``.
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
        elif key == "bench" and isinstance(value, dict) and isinstance(merged.get("bench"), dict):
            bench: dict[str, Any] = dict(merged["bench"])
            for bench_key, bench_value in value.items():
                if (
                    bench_key == "reviewers"
                    and isinstance(bench_value, dict)
                    and isinstance(bench.get("reviewers"), dict)
                ):
                    reviewers: dict[str, Any] = dict(bench["reviewers"])
                    for reviewer_name, reviewer_table in bench_value.items():
                        if isinstance(reviewer_table, dict) and isinstance(reviewers.get(reviewer_name), dict):
                            reviewers[reviewer_name] = {**reviewers[reviewer_name], **reviewer_table}
                        else:
                            reviewers[reviewer_name] = reviewer_table
                    bench["reviewers"] = reviewers
                else:
                    bench[bench_key] = bench_value
            merged["bench"] = bench
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


def load_file_config(root: Path) -> DaydreamFileConfig:
    """Load and merge daydream file configuration from a repo root.

    Reads ``root/pyproject.toml`` ``[tool.daydream]`` (low precedence) and
    ``root/.daydream.toml`` (high precedence), merging the two per-key. The
    dotfile wins on conflicting scalar keys; phase sub-tables merge so each
    source contributes its own phases.

    Args:
        root: Repository root directory to search for config files.

    Returns:
        A ``DaydreamFileConfig``. Absent files yield an empty config.

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

    model = merged.get("model")
    backend = merged.get("backend")
    # shallow_fanout_threshold: tolerate non-int (stray string, list, etc.) by
    # degrading to None rather than crashing the loader. ``0`` is a meaningful
    # value (disable the short-circuit) so it must round-trip unchanged.
    raw_threshold = merged.get("shallow_fanout_threshold")
    threshold: int | None
    if isinstance(raw_threshold, bool):
        # bool is a subclass of int but never a meaningful threshold value.
        threshold = None
    elif isinstance(raw_threshold, int):
        threshold = raw_threshold
    else:
        threshold = None
    # precision_mode: bool only. Any non-bool value (str, int, list) degrades to
    # None rather than crashing the loader; truthy ints are *not* coerced to True
    # so an accidental ``precision_mode = 1`` is treated as unset, not enabled.
    raw_precision = merged.get("precision_mode")
    precision: bool | None = raw_precision if isinstance(raw_precision, bool) else None
    # Per-file-group fix budgets (#201): tolerate junk by degrading to None (the
    # config.py default then applies). bool is excluded even though it subclasses
    # int/float — ``group_max_serial_items = true`` is never a meaningful count.
    return DaydreamFileConfig(
        model=str(model) if model is not None else None,
        backend=str(backend) if backend is not None else None,
        phases=_coerce_phases(merged.get("phases")),
        bench=dict(merged["bench"]) if isinstance(merged.get("bench"), dict) else {},
        shallow_fanout_threshold=threshold,
        precision_mode=precision,
        group_max_wall_s=_coerce_float(merged.get("group_max_wall_s")),
        group_max_serial_items=_coerce_int(merged.get("group_max_serial_items")),
    )

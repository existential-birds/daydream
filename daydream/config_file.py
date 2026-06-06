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
    """

    model: str | None = None
    backend: str | None = None
    phases: dict[str, dict[str, str]] = field(default_factory=dict)

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
    return DaydreamFileConfig(
        model=str(model) if model is not None else None,
        backend=str(backend) if backend is not None else None,
        phases=_coerce_phases(merged.get("phases")),
    )

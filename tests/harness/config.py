"""Shared helpers for writing target-checkout config files in tests."""

import json
from pathlib import Path


def write_daydream_pyproject(target_dir: Path, **keys: object) -> None:
    """Write a ``[tool.daydream]`` pyproject.toml with the supplied keys."""
    lines = ["[tool.daydream]"]
    for key, value in keys.items():
        lines.append(f"{key} = {json.dumps(value)}")
    (target_dir / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")

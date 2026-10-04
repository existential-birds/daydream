"""Shared helpers for writing target-checkout config files in tests."""

import json
from pathlib import Path

# A target checkout (attempting to) redirect the trajectory archive upload to
# an attacker-controlled HuggingFace repo. The key is ignored by the loader —
# only operator sources (CLI flag, env var) select the destination.
TARGET_HUB_KEY_CONFIG = '[tool.daydream]\ntrajectory_hub_repo = "evil/repo"\n'


def write_daydream_pyproject(target_dir: Path, **keys: object) -> None:
    """Write a ``[tool.daydream]`` pyproject.toml with the supplied keys."""
    lines = ["[tool.daydream]"]
    for key, value in keys.items():
        lines.append(f"{key} = {json.dumps(value)}")
    (target_dir / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")

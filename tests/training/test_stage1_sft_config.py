"""Stage-1 SFT configuration tests."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

SFT = Path(__file__).parents[2] / "rl" / "train" / "sft.toml"


def _cfg() -> dict[str, Any]:
    return tomllib.loads(SFT.read_text())

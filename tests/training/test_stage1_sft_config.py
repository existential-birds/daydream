"""Stage-1 SFT configuration tests."""
from __future__ import annotations

import pathlib
import subprocess
import tomllib
from pathlib import Path
from typing import Any

SFT = Path(__file__).parents[2] / "rl" / "train" / "sft.toml"


def _cfg() -> dict[str, Any]:
    return tomllib.loads(SFT.read_text())


def test_dry_run_passes_without_gpu(tmp_path: pathlib.Path, prime_rl_workspace: pathlib.Path) -> None:
    """PATTERN dry-path test: `sft @ <cfg> --dry-run` from inside the prime-rl
    workspace validates every pydantic schema without touching a GPU."""
    # sft.toml is plain prime-rl schema (no daydream-private [dataset] table),
    # so the documented command validates the shipped file directly.
    out_dir = tmp_path / "outputs"
    r = subprocess.run(
        ["uv", "run", "sft", "@", str(SFT), "--dry-run", "--output-dir", str(out_dir)],
        cwd=prime_rl_workspace,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "Dry run complete" in r.stdout
    resolved = out_dir / "configs" / "sft.toml"
    assert resolved.is_file()
    rc = tomllib.loads(resolved.read_text())
    assert rc["model"]["lora"]["rank"] == _cfg()["model"]["lora"]["rank"]
    assert rc["renderer"]["name"] == "default"

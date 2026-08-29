"""Stage-1 dataset-SFT config (rl/train/sft.toml) content + dry-path tests.

Shape per Task 0's spike (plan-notes.md): the dataset-SFT entrypoint is a
separate verb, ``sft @`` — not ``rl @`` — and its schema puts the model block
at top level and the renderer next to it. ``[dataset]`` is a daydream-private
table consumed by our corpus loader (M8/M9), not by prime-rl, so the dry run
validates a copy with that table stripped.
"""

from __future__ import annotations
import pathlib

import subprocess
import tomllib
from typing import Any
from pathlib import Path

SFT = Path(__file__).parents[2] / "rl" / "train" / "sft.toml"


def _cfg() -> dict[str, Any]:
    return tomllib.loads(SFT.read_text())


def _cfg_text() -> str:
    return SFT.read_text()


def test_bf16_lora_in_staged_band() -> None:
    c = _cfg()
    assert c["model"]["optimization_dtype"] == "bfloat16"
    assert c["model"]["reduce_dtype"] == "bfloat16"
    assert 64 <= c["model"]["lora"]["rank"] <= 128
    # 2:1 alpha:rank band.
    assert c["model"]["lora"]["alpha"] == 2.0 * c["model"]["lora"]["rank"]


def test_lora_targets_default_projection_set() -> None:
    # C2 requires prime-rl's default list, so the recipe omits `target_modules`
    # rather than restating it (mirrors rl/train/rl.toml).
    assert "target_modules" not in _cfg()["model"]["lora"]


def test_default_renderer_not_stock_qwen3() -> None:
    # The stock qwen3 renderer injects an empty <think></think> block, which
    # corrupts cross-entropy on completions.
    assert _cfg()["renderer"]["name"] == "default"


def test_seq_len_matches_stage3() -> None:
    assert _cfg()["model"]["seq_len"] == 32768


def test_no_live_teacher_algo_block() -> None:
    # M10: dataset SFT never routes through the live-teacher variant. There is
    # no orchestrator at all on the `sft @` entrypoint, and no [algo] block.
    assert "orchestrator" not in _cfg()
    assert "algo" not in _cfg()


def test_gold_positive_only_dataset_gate() -> None:
    c = _cfg()
    assert c["dataset"]["gold_positive_only"] is True  # M8
    assert c["dataset"]["tier_counts_reported"] is True  # M9: gold vs silver reported separately
    # M23/M9: silver process traces are admitted only in the separately tagged
    # section, never mixed into the gold-positive training data.
    assert "silver" in c["dataset"]
    assert "rubric" in c["dataset"]["silver"]


def test_dry_run_passes_without_gpu(tmp_path: pathlib.Path, prime_rl_workspace: pathlib.Path) -> None:
    """PATTERN dry-path test: `sft @ <cfg> --dry-run` from inside the prime-rl
    workspace validates every pydantic schema without touching a GPU."""
    # Strip the daydream-private [dataset] table (prime-rl forbids extra keys);
    # it is placed last in the file exactly so this copy stays trivial.
    text = _cfg_text()
    marker = "\n[dataset]"
    idx = text.find(marker)
    assert idx != -1, "sft.toml must keep the [dataset] section last for the dry-run copy"
    sanitized = tmp_path / "sft-dryrun.toml"
    sanitized.write_text(text[:idx] + "\n")
    out_dir = tmp_path / "outputs"
    r = subprocess.run(
        ["uv", "run", "sft", "@", str(sanitized), "--dry-run", "--output-dir", str(out_dir)],
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


def test_dry_run_copy_is_valid_toml(tmp_path: pathlib.Path) -> None:
    """The sanitizer the dry-run test relies on must see a valid document."""
    tomllib.loads(_cfg_text())

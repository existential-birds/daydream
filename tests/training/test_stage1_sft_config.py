"""Stage-1 dataset-SFT config (rl/train/sft.toml) content + dry-path tests.

Shape per Task 0's spike (plan-notes.md): the dataset-SFT entrypoint is a
separate verb, ``sft @`` — not ``rl @`` — and its schema puts the model block
at top level and the renderer next to it. ``[dataset]`` is a daydream-private
table consumed by our corpus loader (M8/M9), not by prime-rl, so the dry run
validates a copy with that table stripped.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tomllib
from pathlib import Path
from typing import Any

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
    # M8/M9: the gold-positive-only guarantee lives in the coordinator's Stage-1
    # materialization, NOT a prime-rl-unreadable [dataset] table. sft.toml is
    # plain prime-rl schema so `uv run sft @ sft.toml --dry-run` validates
    # directly with no stripping (issues 7/16).
    assert "dataset" not in _cfg()


def test_coordinator_stage1_materializes_gold_positive_prompt_completion(
    tmp_path: pathlib.Path,
) -> None:
    """M8/M9: the Stage-1 dataset is gold-positive only with tier counts.

    The coordinator writes only accepted-class gold completions as
    prompt/completion JSONL (the prime-rl `sft` loader's accepted shape) and
    reports gold vs silver counts separately in the stage manifest.
    """
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    fixture = Path(__file__).parents[2] / "tests" / "fixtures" / "training" / "records-50"
    run_pipeline(
        PipelineConfig(corpus=fixture / "records.jsonl", out_dir=tmp_path, stages=("stage0", "stage1")),
        dry_run=False,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "stage1" / "sft-dataset.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows  # gold-positive rows were materialized
    # prompt/completion shape — the SFT-loader column contract (issue 5)
    assert all(set(row) == {"prompt", "completion"} for row in rows)
    # M8: no rejected row leaks into the SFT dataset; the fixture's 15 rejected
    # rows (noise chatter) are excluded.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    tier_counts = manifest["stages"]["stage1"]["tier_counts"]
    assert tier_counts["gold"] == len(rows) == 35
    assert tier_counts["silver"] == 0


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


def test_dry_run_copy_is_valid_toml(tmp_path: pathlib.Path) -> None:
    """The recipe must parse cleanly as TOML with no extra-key sections."""
    tomllib.loads(_cfg_text())

"""Price resolution in the offline benchmark report generator (bench/benchmark-report/build.py)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

BUILD_PY = Path(__file__).resolve().parents[1] / "bench" / "benchmark-report" / "build.py"


@pytest.fixture(scope="module")
def build_mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_report_build", BUILD_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PR_URL = "https://github.com/calcom/cal.com/pull/10600"


def _corpus(root: Path) -> argparse.Namespace:
    """A minimal one-PR, one-judge corpus plus one trajectory with known token counts."""
    judge = root / "results" / "anthropic_claude-opus-4-5-20251101"
    judge.mkdir(parents=True)
    leaf = {"tp": 1, "fp": 0, "fn": 0, "total_candidates": 1, "total_golden": 1}
    (judge / "evaluations.json").write_text(json.dumps({PR_URL: {"daydream-owl-alpha": leaf}}))

    traj = root / "trajectories"
    traj.mkdir()
    (traj / "cal.com-10600.json").write_text(
        json.dumps(
            {
                "final_metrics": {
                    "total_prompt_tokens": 1_000_000,
                    "total_completion_tokens": 1_000_000,
                    "total_cached_tokens": 1_000_000,
                    "total_steps": 3,
                }
            }
        )
    )
    return argparse.Namespace(
        results_root=str(root / "results"),
        daydream_tool="daydream-owl-alpha",
        exclude_tool="daydream-glm",
        price_model="glm-5.2",
        trajectories=str(traj),
        pr_labels="",
        dashboard="",
        speed_analysis="",
    )


def test_price_card_comes_from_shared_pricing_table(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no override file, prices resolve from daydream/pricing.py and meta says so."""
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    report: dict[str, Any] = build_mod.build(_corpus(tmp_path))

    assert report["economy"]["price_card"] == {"input": 1.40, "cached": 0.26, "output": 4.40}
    assert report["meta"]["price_source"] == "daydream/pricing.py"
    assert report["economy"]["total_cost_usd"] == pytest.approx(1.40 + 0.26 + 4.40)


def test_prices_file_override_changes_synthesized_cost(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """$DAYDREAM_PRICES_FILE overrides the card, the synthesized cost, and meta.price_source."""
    prices_file = tmp_path / "prices.toml"
    prices_file.write_text(
        '[prices."glm-5.2"]\ninput = 10.0\ncached_input = 2.0\noutput = 20.0\n'
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))
    report: dict[str, Any] = build_mod.build(_corpus(tmp_path))

    assert report["economy"]["price_card"] == {"input": 10.0, "cached": 2.0, "output": 20.0}
    assert report["economy"]["total_cost_usd"] == pytest.approx(32.0)
    assert report["per_pr"][0]["cost_usd"] == pytest.approx(32.0)
    assert report["meta"]["price_source"] == "user price override ($DAYDREAM_PRICES_FILE / ~/.daydream/prices.toml)"


def test_unknown_price_model_is_rejected(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    args = _corpus(tmp_path)
    args.price_model = "no-such-model"
    with pytest.raises(SystemExit, match="unknown --price-model"):
        build_mod.build(args)

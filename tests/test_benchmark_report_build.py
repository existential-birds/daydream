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


def _corpus(
    root: Path,
    *,
    pr_trajectories: dict[str, tuple[str, str | None, int, int, int, int, tuple[str, str] | None]] | None = None,
) -> argparse.Namespace:
    """Minimal corpus. pr_trajectories maps PR url -> (filename, pr_repo, prompt,
    completion, cached, steps, (start_iso, end_iso) | None). Omitted -> the existing
    one-PR legacy corpus (cal.com-10600.json with no pr_repo)."""
    if pr_trajectories is None:
        pr_trajectories = {PR_URL: ("cal.com-10600.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None)}
    judge = root / "results" / "anthropic_claude-opus-4-5-20251101"
    judge.mkdir(parents=True)
    leaf = {"tp": 1, "fp": 0, "fn": 0, "total_candidates": 1, "total_golden": 1}
    (judge / "evaluations.json").write_text(
        json.dumps({pr: {"daydream-owl-alpha": leaf} for pr in pr_trajectories})
    )
    traj = root / "trajectories"
    traj.mkdir()
    for fname, pr_repo, prompt, completion, cached, steps, stamps in pr_trajectories.values():
        body: dict[str, Any] = {
            "final_metrics": {
                "total_prompt_tokens": prompt,
                "total_completion_tokens": completion,
                "total_cached_tokens": cached,
                "total_steps": steps,
            }
        }
        extra: dict[str, Any] = {}
        if pr_repo is not None:
            extra["pr_repo"] = pr_repo
        if stamps is not None:
            extra["phase_events"] = [{"timestamp": s} for s in stamps]
        if extra:
            body["extra"] = extra
        (traj / fname).write_text(json.dumps(body))
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


def test_report_joins_trajectories_by_full_repository_identity(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two same-basename PRs (alpha/widgets vs beta/widgets, PR 7) each get their own metrics."""
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    args = _corpus(
        tmp_path,
        pr_trajectories={
            "https://github.com/alpha/widgets/pull/7": (
                "alpha_widgets-7.json", "alpha/widgets", 1_000_000, 2_000_000, 3_000_000, 5,
                ("2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"),
            ),
            "https://github.com/beta/widgets/pull/7": (
                "beta_widgets-7.json", "beta/widgets", 4_000_000, 5_000_000, 6_000_000, 9,
                ("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"),
            ),
        },
    )
    report = build_mod.build(args)

    rows = {r["pr_url"]: r for r in report["per_pr"]}
    assert len(rows) == 2
    alpha = rows["https://github.com/alpha/widgets/pull/7"]
    beta = rows["https://github.com/beta/widgets/pull/7"]
    assert alpha["prompt_tokens"] == 1_000_000
    assert alpha["completion_tokens"] == 2_000_000
    assert alpha["cached_tokens"] == 3_000_000
    assert alpha["steps"] == 5
    assert alpha["cost_usd"] == pytest.approx(1.4 + 3_000_000 * 0.26 / 1e6 + 2_000_000 * 4.4 / 1e6)
    assert alpha["wall_seconds"] == pytest.approx(120.0)
    assert beta["prompt_tokens"] == 4_000_000
    assert beta["completion_tokens"] == 5_000_000
    assert beta["cached_tokens"] == 6_000_000
    assert beta["steps"] == 9
    assert beta["cost_usd"] == pytest.approx(4_000_000 * 1.4 / 1e6 + 6_000_000 * 0.26 / 1e6 + 5_000_000 * 4.4 / 1e6)
    assert beta["wall_seconds"] == pytest.approx(60.0)

    eco = report["economy"]
    assert eco["n_with_trajectory"] == 2
    assert eco["total_prompt_tokens"] == 5_000_000
    assert eco["total_completion_tokens"] == 7_000_000
    assert eco["total_cached_tokens"] == 9_000_000
    assert eco["total_cost_usd"] == pytest.approx(
        1.4 + 3_000_000 * 0.26 / 1e6 + 2_000_000 * 4.4 / 1e6
        + 4_000_000 * 1.4 / 1e6 + 6_000_000 * 0.26 / 1e6 + 5_000_000 * 4.4 / 1e6
    )
    assert eco["n_with_wall"] == 2


def test_report_rejects_ambiguous_legacy_trajectory_fallback(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy basename key shared by two PRs raises a fail-closed SystemExit."""
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    args = _corpus(
        tmp_path,
        pr_trajectories={
            "https://github.com/alpha/widgets/pull/7": ("widgets-7.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None),
            "https://github.com/beta/widgets/pull/7": ("widgets-7.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None),
        },
    )
    with pytest.raises(SystemExit, match="ambiguous legacy trajectory key 'widgets/7'"):
        build_mod.build(args)

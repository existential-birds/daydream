"""Price resolution and evidence-derived improvement recommendations in the offline benchmark report generator (bench/benchmark-report/build.py)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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

SECOND_PR_URL = "https://github.com/calcom/cal.com/pull/10601"
_COMPLETE_SAAS_TOOLS = ("saas-alpha", "saas-beta", "saas-delta", "saas-gamma", "saas-zeta")
_JUDGE_DIRNAME = "anthropic_claude-opus-4-5-20251101"


def _leaf(tp: int = 1, fp: int = 0, fn: int = 0,
          total_candidates: int = 1, total_golden: int = 1) -> dict[str, int]:
    return {"tp": tp, "fp": fp, "fn": fn,
            "total_candidates": total_candidates, "total_golden": total_golden}


def _tools(n: int, daydream: dict[str, int] | None = None) -> dict[str, dict]:
    tools = {f"saas-{i}": _leaf(tp=1, fp=0) for i in range(n)}
    if daydream is not None:
        tools["daydream-owl-alpha"] = daydream
    return tools


_ANCHOR = "anthropic_claude-opus-4-5-20251101"  # -> display "Opus 4.5"


def _corpus(
    root: Path,
    *,
    pr_trajectories: dict[str, tuple[str, str | None, int, int, int, int, tuple[str, str] | None]] | None = None,
    judges: dict[str, dict[str, dict]] | None = None,
    labels: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Minimal corpus. pr_trajectories maps PR url -> (filename, pr_repo, prompt,
    completion, cached, steps, (start_iso, end_iso) | None). Omitted -> the existing
    one-PR legacy corpus (cal.com-10600.json with no pr_repo)."""
    if pr_trajectories is None:
        pr_trajectories = {PR_URL: ("cal.com-10600.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None)}
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    if judges is None:
        judges = {_ANCHOR: _tools(5, _leaf(tp=1, fp=0))}
    for dirname, tools in judges.items():
        jdir = results_root / dirname
        jdir.mkdir(parents=True, exist_ok=True)
        (jdir / "evaluations.json").write_text(
            json.dumps({pr: dict(tools) for pr in pr_trajectories})
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
    ns = argparse.Namespace(
        results_root=str(results_root),
        daydream_tool="daydream-owl-alpha",
        exclude_tool="daydream-glm",
        price_model="glm-5.2",
        trajectories=str(traj),
        pr_labels="",
        dashboard="",
        speed_analysis="",
    )
    if labels is not None:
        labels_path = results_root / "pr_labels.json"
        labels_path.write_text(json.dumps(labels))
        ns.pr_labels = str(labels_path)
    return ns


def _comparison_corpus(root: Path, incomplete_leaf: dict[str, Any] | None) -> argparse.Namespace:
    """Two-PR corpus: daydream + five fully-covered SaaS tools on both PRs, plus one
    incomplete tool (``saas-incomplete``) present only on the first PR. The second PR's
    leaf for it is ``incomplete_leaf``: None (absent) or ``{"skipped": True}`` (skipped).
    Reuses _corpus for the judge dir + trajectories, then overwrites evaluations.json."""
    args = _corpus(
        root,
        pr_trajectories={
            PR_URL: ("cal.com-10600.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None),
            SECOND_PR_URL: ("cal.com-10601.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None),
        },
    )
    dd_leaf = {"tp": 1, "fp": 0, "fn": 0, "total_candidates": 1, "total_golden": 1}
    complete_leaf = {"tp": 1, "fp": 1, "fn": 1, "total_candidates": 1, "total_golden": 1}
    pr1 = {"daydream-owl-alpha": dd_leaf}
    pr2 = {"daydream-owl-alpha": dd_leaf}
    for tool in _COMPLETE_SAAS_TOOLS:
        pr1[tool] = complete_leaf
        pr2[tool] = complete_leaf
    pr1["saas-incomplete"] = dd_leaf
    if incomplete_leaf is not None:
        pr2["saas-incomplete"] = incomplete_leaf
    judge = root / "results" / _JUDGE_DIRNAME
    (judge / "evaluations.json").write_text(json.dumps({PR_URL: pr1, SECOND_PR_URL: pr2}))
    return args


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


@pytest.mark.parametrize("incomplete_leaf", [None, {"skipped": True}], ids=["missing", "skipped"])
def test_build_excludes_incomplete_saas_tools_from_field_and_rank(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    incomplete_leaf: dict[str, Any] | None,
) -> None:
    """A SaaS tool measured on fewer than the full daydream subset is omitted from the
    field and rank denominator; every admitted tool is fully covered. Regression for #382."""
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    report: dict[str, Any] = build_mod.build(_comparison_corpus(tmp_path, incomplete_leaf))

    judge = next(j for j in report["judges"] if j["id"] == "claude-opus-4-5-20251101")
    field_tools = {r["tool"] for r in judge["field"]}
    assert field_tools == set(_COMPLETE_SAAS_TOOLS)
    assert "saas-incomplete" not in field_tools
    assert {r["n_prs"] for r in judge["field"]} == {2}
    assert judge["subset_pr_count"] == 2
    assert judge["ranks"]["f1"] == (1, 6)
    assert report["meta"]["subset_pr_count"] == 2


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
            "https://github.com/alpha/widgets/pull/7": (
                "widgets-7.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None
            ),
            "https://github.com/beta/widgets/pull/7": (
                "widgets-7.json", None, 1_000_000, 1_000_000, 1_000_000, 3, None
            ),
        },
    )
    with pytest.raises(SystemExit, match="ambiguous legacy trajectory key 'widgets/7'"):
        build_mod.build(args)


def test_report_allows_mixed_canonical_and_unique_legacy_trajectories(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical PR plus a same-basename PR resolving via a UNIQUE legacy trajectory must not crash.

    Regression for the legacy-guard false positive: pre-fix, the share counter
    counted every PR (including ones resolving canonically), so a single legacy
    trajectory shared its basename with a canonical PR and failed closed despite
    the legacy key being unambiguous.
    """
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    args = _corpus(
        tmp_path,
        pr_trajectories={
            # Resolves canonically via extra.pr_repo.
            "https://github.com/alpha/widgets/pull/7": (
                "alpha_widgets-7.json", "alpha/widgets", 1_000_000, 1_000_000, 1_000_000, 3, None
            ),
            # Same basename, but only a unique legacy trajectory (no pr_repo).
            "https://github.com/beta/widgets/pull/7": (
                "widgets-7.json", None, 4_000_000, 5_000_000, 6_000_000, 9, None
            ),
        },
    )
    report = build_mod.build(args)  # must NOT raise SystemExit

    rows = {r["pr_url"]: r for r in report["per_pr"]}
    assert len(rows) == 2
    alpha = rows["https://github.com/alpha/widgets/pull/7"]
    beta = rows["https://github.com/beta/widgets/pull/7"]
    assert alpha["prompt_tokens"] == 1_000_000
    assert alpha["steps"] == 3
    assert beta["prompt_tokens"] == 4_000_000
    assert beta["completion_tokens"] == 5_000_000
    assert beta["cached_tokens"] == 6_000_000
    assert beta["steps"] == 9

    eco = report["economy"]
    assert eco["n_with_trajectory"] == 2
    assert eco["total_prompt_tokens"] == 5_000_000
    assert eco["total_completion_tokens"] == 6_000_000
    assert eco["total_cached_tokens"] == 7_000_000
    assert eco["total_cost_usd"] == pytest.approx(
        1.4 + 1_000_000 * 0.26 / 1e6 + 1_000_000 * 4.4 / 1e6
        + 4_000_000 * 1.4 / 1e6 + 6_000_000 * 0.26 / 1e6 + 5_000_000 * 4.4 / 1e6
    )


TEMPLATE_HTML = BUILD_PY.with_name("template.html")
# All six literals live only inside the old renderImprovements() body; the
# generated index.html must contain none of them. "15–25%" uses an EN-DASH.
_REMOVED_LITERALS = (
    "grafana singleflight",
    "step2_5_dedup_candidates",
    "Sonnet 4.5",
    "GPT-5.2",
    "15\u201325%",
    "bottom ~40%",
)


def _recommendation_case(tmp_path: Path, case: str) -> tuple[dict | None, dict | None, list[dict]]:
    """Return (judges, labels, expected_improvements) for one evidence shape."""
    anchor_src = str(tmp_path / "results" / _ANCHOR / "evaluations.json")
    if case == "no-evidence":
        return None, None, []
    if case == "aggregate-fp":
        judges = {_ANCHOR: _tools(5, _leaf(tp=1, fp=2))}
        return judges, None, [{
            "priority": 1,
            "heading": "Reduce daydream (owl-alpha) false positives under Opus 4.5",
            "body": "daydream produced 2 FP for 1 TP under Opus 4.5 (claude-opus-4-5-20251101).",
            "measurement": "Next-run target: FP below 2, TP at or above 1, on the same 1-PR subset.",
            "citation": f"source: {anchor_src}",
        }]
    if case == "label-slice":
        judges = {_ANCHOR: _tools(5, _leaf(tp=1, fp=3))}
        labels = {PR_URL: {"derived": {"language": "python"}}}
        labels_src = str(tmp_path / "results" / "pr_labels.json")
        return judges, labels, [
            {
                "priority": 1,
                "heading": "Reduce daydream (owl-alpha) false positives under Opus 4.5",
                "body": "daydream produced 3 FP for 1 TP under Opus 4.5 (claude-opus-4-5-20251101).",
                "measurement": "Next-run target: FP below 3, TP at or above 1, on the same 1-PR subset.",
                "citation": f"source: {anchor_src}",
            },
            {
                "priority": 2,
                "heading": "Tighten the noisiest label slice: Language = python",
                "body": "The python language cohort carries 3 FP for 1 TP across 1 PRs.",
                "measurement": "Next-run target: FP below 3 on python language slices.",
                "citation": f"source: {labels_src} (Slices panel)",
            },
        ]
    if case == "missing-judge":
        judges = {
            _ANCHOR: _tools(5, _leaf(tp=1, fp=0)),
            "custom-reviewer": _tools(5),
        }
        return judges, None, [{
            "priority": 3,
            "heading": "Re-judge daydream under custom-reviewer",
            "body": "daydream has no leaf under: custom-reviewer.",
            "measurement": "Next-run target: fill the cross-judge panels and confirm the precision story is judge-robust.",
            "citation": "source: discovered judges custom-reviewer",
        }]
    raise AssertionError(f"unknown case {case!r}")


@pytest.mark.parametrize("case", ["no-evidence", "aggregate-fp", "label-slice", "missing-judge"])
def test_improvements_derive_from_corpus_evidence(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Recommendations derive only from current-run evidence, in priority order."""
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(tmp_path / "absent.toml"))
    judges, labels, expected = _recommendation_case(tmp_path, case)
    report: dict[str, Any] = build_mod.build(_corpus(tmp_path, judges=judges, labels=labels))
    assert report["improvements"] == expected

"""Tests for the unified bench report builder (cost synthesis + report shape)."""

from __future__ import annotations

import json
from pathlib import Path

from daydream.benchmark.report import PRRun, build_report, synthesize_cost, write_report

#: A price card with round numbers so the two cost branches are distinguishable
#: by inspection: 1M fresh input = $1.00, 1M cached input = $0.10.
_PRICES_TOML = """
[prices."test-model"]
input = 1.0
cached_input = 0.1
output = 2.0
"""


def _install_price_card(tmp_path: Path, monkeypatch) -> None:
    """Point DAYDREAM_PRICES_FILE at a card for ``test-model`` (the D5 override seam)."""
    prices = tmp_path / "prices.toml"
    prices.write_text(_PRICES_TOML, encoding="utf-8")
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices))


def test_pi_cost_synthesis_uses_disjoint_counters(tmp_path, monkeypatch):
    """pi reports fresh input and cache reads as DISJOINT counters — bill both."""
    _install_price_card(tmp_path, monkeypatch)

    cost = synthesize_cost(
        backend="pi",
        model="test-model",
        prompt_tokens=1_000_000,
        cached_tokens=1_000_000,
        completion_tokens=0,
    )

    # 1M fresh × $1.00 + 1M cached × $0.10. NOT $0.10, which is what
    # subtracting cached from prompt (the non-pi rule) would produce.
    assert cost == 1.10
    assert cost != 0.10


def test_non_pi_cost_synthesis_subtracts_cached_from_total(tmp_path, monkeypatch):
    """Claude/Codex report cached input as a subset of the prompt total."""
    _install_price_card(tmp_path, monkeypatch)

    cost = synthesize_cost(
        backend="codex",
        model="test-model",
        prompt_tokens=1_000_000,
        cached_tokens=1_000_000,
        completion_tokens=0,
    )

    # Fresh input is (1M − 1M) = 0, so only the cached read is billed.
    assert cost == 0.10


def test_measured_cost_wins_over_synthesis(tmp_path, monkeypatch):
    """A non-zero trajectory ``total_cost_usd`` beats any synthesized figure."""
    _install_price_card(tmp_path, monkeypatch)
    trajectory = tmp_path / "acme_widgets-1.json"
    trajectory.write_text(
        json.dumps(
            {
                "final_metrics": {
                    "total_prompt_tokens": 1_000_000,
                    "total_cached_tokens": 1_000_000,
                    "total_completion_tokens": 0,
                    "total_cost_usd": 7.25,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        corpus="harvested",
        corpus_root=tmp_path,
        tool_label="daydream",
        reviewer_backend="pi",
        reviewer_model="test-model",
        reviewer_provider=None,
        judge_route="anthropic-direct",
        judge_model=None,
        git_sha="abc123",
        timestamp="2026-07-18T00:00:00+00:00",
        pr_runs=[
            PRRun(
                golden_url="https://github.com/acme/widgets/pull/1",
                injected_comments=3,
                elapsed_s=12.5,
                trajectory_path=trajectory,
            )
        ],
    )

    entry = report["prs"][0]
    assert entry["cost_source"] == "measured"
    assert entry["cost_usd"] == 7.25  # not the 1.10 the pi counters would synthesize
    assert entry["prompt_tokens"] == 1_000_000
    assert entry["cached_tokens"] == 1_000_000
    assert entry["elapsed_s"] == 12.5


def test_non_finite_metrics_fall_back_instead_of_crashing(tmp_path, monkeypatch):
    """json.loads accepts NaN/Infinity — they must not reach int()/the report."""
    _install_price_card(tmp_path, monkeypatch)
    trajectory = tmp_path / "acme_widgets-2.json"
    trajectory.write_text(
        '{"final_metrics": {"total_prompt_tokens": Infinity, '
        '"total_cached_tokens": 1000000, "total_completion_tokens": NaN, '
        '"total_cost_usd": NaN}}',
        encoding="utf-8",
    )

    report = build_report(
        corpus="harvested",
        corpus_root=tmp_path,
        tool_label="daydream",
        reviewer_backend="pi",
        reviewer_model="test-model",
        reviewer_provider=None,
        judge_route="anthropic-direct",
        judge_model=None,
        git_sha="abc123",
        timestamp="2026-07-18T00:00:00+00:00",
        pr_runs=[
            PRRun(
                golden_url="https://github.com/acme/widgets/pull/2",
                injected_comments=1,
                elapsed_s=1.0,
                trajectory_path=trajectory,
            )
        ],
    )

    entry = report["prs"][0]
    assert entry["prompt_tokens"] == 0
    assert entry["completion_tokens"] == 0
    assert entry["cached_tokens"] == 1_000_000
    # NaN cost is not "measured": it falls through to synthesis from the
    # surviving counters (1M cached × $0.10 under the pi branch).
    assert entry["cost_source"] == "synthesized"
    assert entry["cost_usd"] == 0.10
    assert json.dumps(report, allow_nan=False)


def test_missing_trajectory_yields_unknown_cost_without_raising(tmp_path):
    """A PR whose trajectory never landed still gets a well-formed entry."""
    report = build_report(
        corpus="withmartian",
        corpus_root=tmp_path,
        tool_label="daydream",
        reviewer_backend=None,
        reviewer_model=None,
        reviewer_provider=None,
        judge_route="martian",
        judge_model=None,
        git_sha="abc123",
        timestamp="2026-07-18T00:00:00+00:00",
        pr_runs=[
            PRRun(
                golden_url="https://github.com/acme/widgets/pull/9",
                injected_comments=0,
                elapsed_s=0.0,
                trajectory_path=tmp_path / "does-not-exist.json",
            )
        ],
    )

    entry = report["prs"][0]
    assert entry["cost_usd"] is None and entry["cost_source"] == "unknown"
    assert entry["prompt_tokens"] == 0 and entry["completion_tokens"] == 0


def _report_for(corpus: str, tmp_path: Path) -> dict:
    return build_report(
        corpus=corpus,
        corpus_root=tmp_path / corpus,
        tool_label="daydream",
        reviewer_backend="claude",
        reviewer_model="sonnet",
        reviewer_provider=None,
        judge_route="anthropic-direct",
        judge_model="judge-x",
        git_sha="abc123",
        timestamp="2026-07-18T00:00:00+00:00",
        pr_runs=[
            PRRun(
                golden_url=f"https://github.com/acme/{corpus}/pull/1",
                injected_comments=2,
                elapsed_s=1.0,
                trajectory_path=tmp_path / "absent.json",
                score_leaf={"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0},
            )
        ],
        aggregate={
            "scored_pr_count": 1,
            "tp": 1,
            "fp": 1,
            "fn": 0,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2 / 3,
        },
    )


def test_build_report_shape_is_corpus_agnostic(tmp_path):
    """Both corpus kinds produce the same key set at every level (AC1)."""
    withmartian = _report_for("withmartian", tmp_path)
    harvested = _report_for("harvested", tmp_path)

    assert set(withmartian) == set(harvested)
    assert set(withmartian) == {
        "schema_version",
        "corpus",
        "corpus_root",
        "tool_label",
        "reviewer_backend",
        "reviewer_model",
        "reviewer_provider",
        "judge_route",
        "judge_model",
        "git_sha",
        "timestamp",
        "prs",
        "aggregate",
        "distribution",
    }
    assert set(withmartian["prs"][0]) == set(harvested["prs"][0])
    assert set(withmartian["prs"][0]) == {
        "golden_url",
        "trial_index",
        "injected_comments",
        "elapsed_s",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cost_usd",
        "cost_source",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
    }
    assert withmartian["corpus"] == "withmartian" and harvested["corpus"] == "harvested"
    assert withmartian["schema_version"] == 1
    assert set(withmartian["aggregate"]) == set(harvested["aggregate"])
    assert withmartian["prs"][0]["tp"] == 1 and withmartian["prs"][0]["precision"] == 0.5


def test_write_report_creates_parent_dirs_and_round_trips(tmp_path):
    dest = tmp_path / ".daydream-bench" / "report-daydream.json"
    report = _report_for("harvested", tmp_path)

    written = write_report(dest, report)

    assert written == dest
    assert json.loads(dest.read_text(encoding="utf-8")) == report

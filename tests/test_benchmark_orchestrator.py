"""Tests for the benchmark orchestrator (acquire → review → map → inject)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from daydream.benchmark.acquire import AcquiredCheckout
from daydream.benchmark.config import BenchConfig
from daydream.benchmark.orchestrator import run_bench
from daydream.benchmark.prs import load_evaluable_prs
from daydream.benchmark.score import DaydreamScores


def _item(file: str, line: int, **kw: Any) -> dict[str, Any]:
    """Build a merged-item dict with all required fields."""
    item: dict[str, Any] = {
        "id": kw.get("id", f"{file}:{line}"),
        "description": kw.get("description", "A finding"),
        "file": file,
        "line": line,
        "confidence": kw.get("confidence", "high"),
        "rationale": kw.get("rationale", "Because reasons"),
        "lens": kw.get("lens", "correctness"),
        "severity": kw.get("severity", "medium"),
    }
    return item


def _seed_benchmark_data_with_all_26_keys(tmp_path: Path) -> Path:
    """Seed benchmark_data.json with all 26 golden_urls and return its path.

    Places the corpus at ``<benchmark_repo>/results/benchmark_data.json`` per
    the orchestrator's path convention; the benchmark_repo is ``tmp_path``.
    """
    data: dict[str, Any] = {}
    for pr in load_evaluable_prs():
        data[pr.golden_url] = {
            "golden_comments": [{"path": "f.py", "line": 1, "body": "golden"}],
            "reviews": [
                {
                    "tool": "other-tool",
                    "repo_name": "other-tool",
                    "pr_url": pr.golden_url,
                    "review_comments": [],
                }
            ],
        }
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_path = results_dir / "benchmark_data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data_path


def _fake_acquired(tmp_path: Path) -> AcquiredCheckout:
    """Return an AcquiredCheckout over an existing directory with a pinned base."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    return AcquiredCheckout(path=checkout, base_sha="0" * 40)


def _write_items(checkout: Path, items: list[dict[str, Any]]) -> Path:
    """Write {"items": items} to the artifact path and return it."""
    artifact = checkout / ".daydream" / "deep" / "merged-items.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"items": items}), encoding="utf-8")
    return artifact


def _config(tmp_path: Path, data_path: Path, *, score: bool, only: str) -> BenchConfig:
    """Build a BenchConfig whose benchmark_repo derives data_path.

    data_path == benchmark_repo / "results" / "benchmark_data.json", so the
    benchmark_repo is data_path.parent.parent.
    """
    benchmark_repo = data_path.parent.parent
    return BenchConfig(
        benchmark_repo=benchmark_repo,
        cache_dir=tmp_path / "cache",
        force=False,
        score=score,
        only=only,
        limit=None,
        trajectory_dir=tmp_path / "trajectories",
    )


def test_run_bench_injects_a_daydream_review_per_selected_pr(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda checkout, **k: _write_items(checkout, [_item("f.py", 1)]),
    )
    rc = run_bench(_config(tmp_path, data_path, score=False, only="grafana"))  # 10 PRs
    data = json.loads(data_path.read_text())
    grafana = [u for u in data if "grafana" in u]
    assert rc == 0 and len(grafana) == 10
    assert all(any(r["tool"] == "daydream" for r in data[u]["reviews"]) for u in grafana)


def test_run_bench_announces_and_reports_each_pr(tmp_path, monkeypatch):
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.benchmark.orchestrator.console", rec)
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda checkout, **k: _write_items(checkout, [_item("f.py", 1)]),
    )
    run_bench(replace(_config(tmp_path, data_path, score=False, only="grafana"), limit=1))
    out = rec.export_text()
    assert "Reviewing" in out and "grafana" in out  # announced before the blocking review
    assert re.search(r"\b\d+s\b", out)  # completion shows elapsed
    assert "1 finding" in out  # finding count for the injected PR


def test_verbose_streams_child_output(tmp_path, monkeypatch):
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.benchmark.orchestrator.console", rec)
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )

    def review(checkout, on_line=None, **k):
        if on_line:
            on_line("CHILD-LINE-XYZ\n")
        return _write_items(checkout, [_item("f.py", 1)])

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_daydream_review", review)
    cfg = replace(_config(tmp_path, data_path, score=False, only="grafana"), limit=1, verbose=True)
    run_bench(cfg)
    assert "CHILD-LINE-XYZ" in rec.export_text()  # verbose forwards child output


def test_orchestrator_forwards_reviewer_fields(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )

    def cap_review(checkout, **k):
        captured.update(k)
        return _write_items(checkout, [_item("f.py", 1)])

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_daydream_review", cap_review)
    cfg = _config(tmp_path, data_path, score=False, only="grafana")
    cfg = replace(cfg, reviewer_backend="pi", reviewer_model="glm-5.2", reviewer_provider="openrouter")
    run_bench(cfg)
    assert (captured["backend"], captured["model"], captured["provider"]) == ("pi", "glm-5.2", "openrouter")


def test_direct_anthropic_preflight_runs_before_review(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = {"review": 0}
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda *a, **k: calls.__setitem__("review", 1),
    )
    cfg = replace(_config(tmp_path, data_path, score=True, only="grafana"), judge_route="anthropic-direct")
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        run_bench(cfg)
    assert calls["review"] == 0


def test_orchestrator_passes_judge_route_to_scoring(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-direct")
    monkeypatch.delenv("MARTIAN_BASE_URL", raising=False)
    monkeypatch.setenv("MARTIAN_MODEL", "claude-opus-4-5-20251101")
    monkeypatch.setattr("daydream.benchmark.orchestrator.acquire_checkout", lambda *a, **k: _fake_acquired(tmp_path))
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda checkout, **k: _write_items(checkout, [_item("f.py", 1)]),
    )
    captured = {}

    def fake_score(repo, model, *, golden_urls, tool, judge_route):
        captured.update(model=model, golden_urls=golden_urls, tool=tool, judge_route=judge_route)
        return DaydreamScores(scored_pr_count=1, total_tp=1, precision=1.0, recall=1.0)

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_scoring", fake_score)
    cfg = replace(_config(tmp_path, data_path, score=True, only="grafana"), limit=1, judge_route="anthropic-direct")
    assert run_bench(cfg) == 0
    assert captured["judge_route"] == "anthropic-direct"
    assert len(captured["golden_urls"]) == 1


def test_force_score_invalidates_stale_scorer_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    golden_url = next(url for url in data if "grafana" in url)
    data[golden_url]["reviews"].append(
        {
            "tool": "daydream",
            "repo_name": "daydream",
            "pr_url": golden_url,
            "review_comments": [{"body": "old candidate"}],
        }
    )
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    scores_dir = tmp_path / "results" / "judge-model"
    scores_dir.mkdir(parents=True)
    for name in ("candidates.json", "dedup_groups.json", "evaluations.json"):
        (scores_dir / name).write_text(
            json.dumps({golden_url: {"daydream": ["stale"], "other-tool": ["keep"]}}, indent=2),
            encoding="utf-8",
        )

    _mock_review(tmp_path, monkeypatch)

    def fake_score(repo, model, *, golden_urls, tool, judge_route):
        for name in ("candidates.json", "dedup_groups.json", "evaluations.json"):
            artifact = json.loads((scores_dir / name).read_text(encoding="utf-8"))
            assert artifact == {golden_url: {"other-tool": ["keep"]}}
        return DaydreamScores(scored_pr_count=1, total_tp=1, precision=1.0, recall=1.0)

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_scoring", fake_score)
    cfg = replace(_config(tmp_path, data_path, score=True, only="grafana"), force=True, limit=1)

    assert run_bench(cfg) == 0


def test_force_without_score_invalidates_artifacts_for_a_later_scoring_run(tmp_path, monkeypatch):
    """Real-path cross-run: run 1 is --force with no --score, run 2 is --score with no
    --force. Run 2 takes the already-injected skip branch, so the stale leaves run 1
    orphaned must already be gone from disk by the time scoring is dispatched.
    """
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    golden_url = next(url for url in data if "grafana" in url)
    data[golden_url]["reviews"].append(
        {
            "tool": "daydream",
            "repo_name": "daydream",
            "pr_url": golden_url,
            "review_comments": [{"body": "old candidate"}],
        }
    )
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    scores_dir = tmp_path / "results" / "judge-model"
    scores_dir.mkdir(parents=True)
    for name in ("candidates.json", "dedup_groups.json", "evaluations.json"):
        (scores_dir / name).write_text(
            json.dumps({golden_url: {"daydream": ["stale"], "other-tool": ["keep"]}}, indent=2),
            encoding="utf-8",
        )

    _mock_review(tmp_path, monkeypatch)

    assert run_bench(replace(_config(tmp_path, data_path, score=False, only="grafana"), force=True, limit=1)) == 0
    for name in ("candidates.json", "dedup_groups.json", "evaluations.json"):
        artifact = json.loads((scores_dir / name).read_text(encoding="utf-8"))
        assert artifact == {golden_url: {"other-tool": ["keep"]}}, f"{name} kept a stale daydream leaf"

    dispatched: dict[str, Any] = {}

    def fake_score(repo, model, *, golden_urls, tool, judge_route):
        dispatched["artifacts"] = {
            name: json.loads((scores_dir / name).read_text(encoding="utf-8"))
            for name in ("candidates.json", "dedup_groups.json", "evaluations.json")
        }
        return DaydreamScores(scored_pr_count=1, total_tp=1, precision=1.0, recall=1.0)

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_scoring", fake_score)
    assert run_bench(replace(_config(tmp_path, data_path, score=True, only="grafana"), limit=1)) == 0
    assert dispatched["artifacts"] == {
        name: {golden_url: {"other-tool": ["keep"]}}
        for name in ("candidates.json", "dedup_groups.json", "evaluations.json")
    }


def test_scoring_ignores_unselected_prs_left_in_evaluations(tmp_path, monkeypatch):
    """Real-path: with --only selecting one PR, leaves that a wider earlier run left
    in the resumable evaluations.json under the same tool label must not be scored.

    Only the judge subprocesses are faked; selection, scoring dispatch, artifact
    read, and parse run for real.
    """
    rec = Console(record=True, force_terminal=True, width=200)
    monkeypatch.setattr("daydream.benchmark.orchestrator.console", rec)
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    selected = next(url for url in data if "grafana" in url)
    stale = next(url for url in data if "grafana" not in url)

    scores_dir = tmp_path / "results" / "judge-model"
    scores_dir.mkdir(parents=True)
    (scores_dir / "evaluations.json").write_text(
        json.dumps(
            {
                selected: {"daydream": {"tp": 1, "fp": 0, "fn": 0, "total_candidates": 1, "total_golden": 1}},
                stale: {"daydream": {"tp": 0, "fp": 9, "fn": 9, "total_candidates": 9, "total_golden": 9}},
            }
        ),
        encoding="utf-8",
    )

    _mock_review(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "daydream.benchmark.score.subprocess.run",
        lambda cmd, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    cfg = replace(_config(tmp_path, data_path, score=True, only="grafana"), limit=1)
    assert run_bench(cfg) == 0

    out = rec.export_text()
    assert "aggregate over 1 PR(s)" in out
    assert stale not in out
    assert "precision=1.000" in out and "recall=1.000" in out


def test_materiality_gate_filters_submission(tmp_path, monkeypatch):
    from daydream.benchmark.orchestrator import _injected_comment_count

    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda checkout, **k: _write_items(
            checkout,
            [
                _item("high.py", 1, id=1, confidence="HIGH"),
                _item("med.py", 2, id=2, confidence="MEDIUM"),
                _item("low.py", 3, id=3, confidence="LOW"),
            ],
        ),
    )
    cfg = replace(
        _config(tmp_path, data_path, score=False, only="grafana"),
        limit=1,
        min_confidence="HIGH",
    )
    assert run_bench(cfg) == 0
    data = json.loads(data_path.read_text())
    [golden_url] = [u for u in data if "grafana" in u and any(r["tool"] == "daydream" for r in data[u]["reviews"])]
    assert _injected_comment_count(data, golden_url, "daydream") == 1  # only the HIGH item survives the gate


def _mock_review(tmp_path, monkeypatch):
    """Wire acquire_checkout + run_daydream_review to a one-finding fake review."""
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda checkout, **k: _write_items(checkout, [_item("f.py", 1)]),
    )


def _scores_by_trial(tmp_path, monkeypatch):
    """Mock run_scoring: derive distinct precision/recall/f1 from the trial label.

    The trial-suffixed tool label (``daydream-t00``…) encodes the trial index, so
    the mock returns a different score per trial — proving the distribution is
    aggregated over genuinely separate trial runs, not one value repeated.
    """
    captured = {"tools": [], "repos": []}

    def fake_score(repo, model, *, golden_urls, tool, judge_route):
        captured["tools"].append(tool)
        captured["repos"].append(repo)
        idx = int(tool.rsplit("-t", 1)[1]) if "-t" in tool else 0
        p = 0.4 + 0.1 * idx
        r = 0.5 + 0.1 * idx
        f1 = 2 * p * r / (p + r)
        return DaydreamScores(
            scored_pr_count=1,
            total_tp=1,
            total_fp=idx,
            total_fn=idx + 1,
            total_errors=0,
            total_comparisons=10 + idx,
            precision=p,
            recall=r,
            f1=f1,
        )

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_scoring", fake_score)
    return captured


def _scored_trials_config(tmp_path, data_path, trials):
    """Build a scoring-enabled BenchConfig with N trials over the grafana subset."""
    return replace(_config(tmp_path, data_path, score=True, only="grafana"), limit=1, trials=trials)


def test_trials_3_creates_3_isolated_trial_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    _mock_review(tmp_path, monkeypatch)
    captured = _scores_by_trial(tmp_path, monkeypatch)

    rc = run_bench(_scored_trials_config(tmp_path, data_path, 3))

    assert rc == 0
    trials_dir = tmp_path / ".daydream-bench" / "trials" / "daydream"
    trial_dirs = sorted(p.name for p in trials_dir.iterdir() if p.name.startswith("trial-"))
    assert trial_dirs == ["trial-00", "trial-01", "trial-02"]
    # Each trial owns its own corpus, scored under its own suffixed label + repo.
    for i in range(3):
        corpus = trials_dir / f"trial-{i:02d}" / "results" / "benchmark_data.json"
        assert corpus.exists()
    assert captured["tools"] == ["daydream-t00", "daydream-t01", "daydream-t02"]
    assert all(r == trials_dir / f"trial-{i:02d}" for i, r in enumerate(captured["repos"]))


def test_trials_injects_into_trial_corpus_not_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    canonical_before = data_path.read_bytes()
    _mock_review(tmp_path, monkeypatch)
    _scores_by_trial(tmp_path, monkeypatch)

    run_bench(_scored_trials_config(tmp_path, data_path, 2))

    # R3: the canonical corpus is byte-for-byte unchanged (no trial keys leak in).
    assert data_path.read_bytes() == canonical_before
    # The trial's own corpus carries the injected trial-labeled review.
    trial0_corpus = (
        tmp_path / ".daydream-bench" / "trials" / "daydream" / "trial-00" / "results" / "benchmark_data.json"
    )
    trial0 = json.loads(trial0_corpus.read_text())
    grafana = [u for u in trial0 if "grafana" in u]
    injected = [u for u in grafana if any(r["tool"] == "daydream-t00" for r in trial0[u]["reviews"])]
    assert len(injected) == 1


def test_trials_writes_summary_with_distribution_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    _mock_review(tmp_path, monkeypatch)
    _scores_by_trial(tmp_path, monkeypatch)

    run_bench(_scored_trials_config(tmp_path, data_path, 3))

    summary_path = tmp_path / ".daydream-bench" / "trials" / "daydream" / "trials-summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["tool_label"] == "daydream"
    assert summary["trials"] == 3
    assert summary["judge_route"] == "martian"
    assert summary["judge_model"] == "judge-model"
    assert isinstance(summary["git_sha"], str) and summary["git_sha"]
    assert summary["timestamp"]
    assert len(summary["pr_set"]) == 1
    assert len(summary["per_trial"]) == 3
    # per-trial precisions are the three distinct values, in trial order.
    assert [round(pt["precision"], 3) for pt in summary["per_trial"]] == [0.4, 0.5, 0.6]
    # Richer per-trial fields carry the tp/fp/fn/comparison counts for post-hoc auditing.
    assert [pt["tool_label"] for pt in summary["per_trial"]] == ["daydream-t00", "daydream-t01", "daydream-t02"]
    assert [pt["total_fp"] for pt in summary["per_trial"]] == [0, 1, 2]
    assert [pt["total_fn"] for pt in summary["per_trial"]] == [1, 2, 3]
    assert [pt["total_comparisons"] for pt in summary["per_trial"]] == [10, 11, 12]
    for pt in summary["per_trial"]:
        assert pt["scored_pr_count"] == 1 and pt["total_tp"] == 1 and pt["total_errors"] == 0
    dist = summary["distribution"]
    for metric in ("precision", "recall", "f1"):
        assert set(dist[metric]) == {"mean", "median", "stddev", "min", "max", "ci_low", "ci_high"}
    assert dist["precision"]["mean"] == pytest.approx(0.5)
    assert dist["precision"]["min"] == pytest.approx(0.4)
    assert dist["precision"]["max"] == pytest.approx(0.6)


def test_trials_prints_distribution_and_cost_estimate(tmp_path, monkeypatch):
    rec = Console(record=True, force_terminal=True, width=120)
    monkeypatch.setattr("daydream.benchmark.orchestrator.console", rec)
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    _mock_review(tmp_path, monkeypatch)
    _scores_by_trial(tmp_path, monkeypatch)

    run_bench(_scored_trials_config(tmp_path, data_path, 3))

    out = rec.export_text()
    assert "Estimated judge cost" in out and "3 trials" in out  # cost surfaced before the loop
    assert "precision" in out and "recall" in out and "f1" in out  # distribution table
    assert "median" in out and "stddev" in out and "ci95" in out
    # Post-loop close on the up-front estimate: 10 + 11 + 12 comparisons across the 3 trials.
    assert "Actual judge comparisons recorded: 33" in out


def test_trials_1_is_backcompat_no_trial_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    _mock_review(tmp_path, monkeypatch)
    captured = _scores_by_trial(tmp_path, monkeypatch)

    rc = run_bench(replace(_config(tmp_path, data_path, score=True, only="grafana"), limit=1, trials=1))

    assert rc == 0
    # trials==1 keeps the legacy path: canonical corpus is written, no sidecar dirs.
    assert not (tmp_path / ".daydream-bench" / "trials").exists()
    data = json.loads(data_path.read_text())
    grafana_injected = [u for u in data if "grafana" in u and any(r["tool"] == "daydream" for r in data[u]["reviews"])]
    assert len(grafana_injected) == 1
    assert captured["tools"] == ["daydream"]  # scored under the base label, not a trial suffix


def test_rerun_skips_already_injected_unless_forced(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    calls = {"n": 0}
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )

    def counting_review(checkout, **k):
        calls["n"] += 1
        return _write_items(checkout, [_item("f.py", 1)])

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_daydream_review", counting_review)
    cfg = _config(tmp_path, data_path, score=False, only="grafana")
    run_bench(cfg)
    first = calls["n"]
    run_bench(cfg)  # force=False
    assert first == 10 and calls["n"] == 10  # second run added zero new reviews


def test_single_shot_run_writes_json_report(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_acquired(tmp_path),
    )
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.run_daydream_review",
        lambda checkout, **k: _write_items(checkout, [_item("f.py", 1)]),
    )

    rc = run_bench(_config(tmp_path, data_path, score=False, only="grafana"))  # 10 PRs

    assert rc == 0
    report_path = tmp_path / ".daydream-bench" / "report-daydream.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["corpus"] == "withmartian"
    assert report["corpus_root"] == str(tmp_path)
    assert report["tool_label"] == "daydream"
    assert len(report["prs"]) == 10
    assert all("grafana" in entry["golden_url"] for entry in report["prs"])
    assert all(entry["injected_comments"] == 1 for entry in report["prs"])
    # Scoring was off, so there is no aggregate and no per-PR score leaf.
    assert report["aggregate"] is None
    assert report["distribution"] is None
    assert all(entry["tp"] is None for entry in report["prs"])
    assert all(entry["trial_index"] is None for entry in report["prs"])  # single-shot carries no trial identity


def test_multi_trial_report_attributes_each_pr_entry_to_its_trial(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-x")
    monkeypatch.setenv("MARTIAN_MODEL", "judge-model")
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    _mock_review(tmp_path, monkeypatch)

    def fake_score(repo, model, *, golden_urls, tool, judge_route):
        idx = int(tool.rsplit("-t", 1)[1])
        return DaydreamScores(
            scored_pr_count=1,
            total_tp=idx,
            total_fp=0,
            total_fn=0,
            total_errors=0,
            total_comparisons=1,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            per_pr={url: {"tp": idx, "fp": 0, "fn": 0} for url in golden_urls},
        )

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_scoring", fake_score)

    assert run_bench(_scored_trials_config(tmp_path, data_path, 3)) == 0

    report = json.loads((tmp_path / ".daydream-bench" / "report-daydream.json").read_text(encoding="utf-8"))
    entries = report["prs"]
    assert len(entries) == 3
    assert len({e["golden_url"] for e in entries}) == 1  # one PR reviewed three times
    assert [e["trial_index"] for e in entries] == [0, 1, 2]
    # Each entry's score leaf tracks its own trial, so leaves stay attributable.
    assert [e["tp"] for e in entries] == [0, 1, 2]

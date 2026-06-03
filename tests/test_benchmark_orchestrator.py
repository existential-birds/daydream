"""Tests for the benchmark orchestrator (acquire → review → map → inject)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.benchmark.config import BenchConfig
from daydream.benchmark.orchestrator import run_bench
from daydream.benchmark.prs import load_evaluable_prs


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


def _fake_checkout(tmp_path: Path) -> Path:
    """Return a Path to an existing checkout directory."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    return checkout


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
        lambda *a, **k: _fake_checkout(tmp_path),
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


def test_rerun_skips_already_injected_unless_forced(tmp_path, monkeypatch):
    data_path = _seed_benchmark_data_with_all_26_keys(tmp_path)
    calls = {"n": 0}
    monkeypatch.setattr(
        "daydream.benchmark.orchestrator.acquire_checkout",
        lambda *a, **k: _fake_checkout(tmp_path),
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

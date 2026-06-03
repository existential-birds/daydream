from types import SimpleNamespace

import pytest

from daydream.benchmark.score import model_results_dir, parse_daydream_scores, preflight_judge_env, run_scoring


def test_model_results_dir_sanitizes_slashes(tmp_path):
    assert model_results_dir(tmp_path, "anthropic/claude-opus-4.5").name == "anthropic_claude-opus-4.5"


def test_preflight_raises_when_key_unset(monkeypatch):
    monkeypatch.delenv("MARTIAN_API_KEY", raising=False)
    with pytest.raises(EnvironmentError) as e:
        preflight_judge_env()
    assert "MARTIAN_API_KEY" in str(e.value)


def test_preflight_passes_when_key_present(monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    preflight_judge_env()


def test_parse_daydream_scores_extracts_per_pr_and_aggregate():
    evals = {
      "url1": {"daydream": {"tp": 2, "fp": 1, "fn": 1, "precision": 0.667, "recall": 0.667,
                            "total_candidates": 3, "total_golden": 3},
               "coderabbit": {"tp": 9, "fp": 0, "fn": 0}},
      "url2": {"daydream": {"tp": 0, "fp": 2, "fn": 3, "precision": 0.0, "recall": 0.0,
                            "total_candidates": 2, "total_golden": 3}}}
    s = parse_daydream_scores(evals)
    assert s.scored_pr_count == 2
    assert s.total_tp == 2 and s.total_fp == 3 and s.total_fn == 4
    assert s.precision == pytest.approx(2 / 5) and s.recall == pytest.approx(2 / 6)
    assert all("coderabbit" not in pr for pr in s.per_pr.values())


def test_run_scoring_invokes_three_steps_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    calls = []
    monkeypatch.setattr("daydream.benchmark.score.subprocess.run",
        lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    rdir = tmp_path / "results" / "anthropic_claude-opus-4.5"
    rdir.mkdir(parents=True)
    (rdir / "evaluations.json").write_text("{}")
    run_scoring(tmp_path, "anthropic/claude-opus-4.5")
    mods = [c[c.index("-m") + 1] for c in calls]
    assert mods == ["code_review_benchmark.step2_extract_comments",
                    "code_review_benchmark.step2_5_dedup_candidates",
                    "code_review_benchmark.step3_judge_comments"]
    assert all(c[c.index("--tool") + 1] == "daydream" for c in calls)

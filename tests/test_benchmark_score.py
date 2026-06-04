from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.benchmark.score import (
    BenchmarkArtifactError,
    JudgeEnvError,
    model_results_dir,
    parse_daydream_scores,
    preflight_judge_env,
    run_scoring,
)


def test_model_results_dir_sanitizes_slashes(tmp_path):
    assert model_results_dir(tmp_path, "anthropic/claude-opus-4.5").name == "anthropic_claude-opus-4.5"


def test_preflight_raises_when_key_unset(monkeypatch):
    monkeypatch.delenv("MARTIAN_API_KEY", raising=False)
    with pytest.raises(JudgeEnvError) as e:
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


def test_run_scoring_passes_limit_to_step3_when_pr_count_given(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    calls = []
    monkeypatch.setattr("daydream.benchmark.score.subprocess.run",
        lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    rdir = tmp_path / "results" / "anthropic_claude-opus-4.5"
    rdir.mkdir(parents=True)
    (rdir / "evaluations.json").write_text("{}")
    run_scoring(tmp_path, "anthropic/claude-opus-4.5", pr_count=3)
    step3_cmd = next(c for c in calls if c[c.index("-m") + 1] == "code_review_benchmark.step3_judge_comments")
    assert "--limit" in step3_cmd
    assert step3_cmd[step3_cmd.index("--limit") + 1] == "3"


def test_run_scoring_omits_limit_from_step3_when_pr_count_not_given(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    calls = []
    monkeypatch.setattr("daydream.benchmark.score.subprocess.run",
        lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    rdir = tmp_path / "results" / "anthropic_claude-opus-4.5"
    rdir.mkdir(parents=True)
    (rdir / "evaluations.json").write_text("{}")
    run_scoring(tmp_path, "anthropic/claude-opus-4.5")
    step3_cmd = next(c for c in calls if c[c.index("-m") + 1] == "code_review_benchmark.step3_judge_comments")
    assert "--limit" not in step3_cmd


def _emulating_fake_run(model_dir_name: str):
    """Build a fake ``subprocess.run`` that emulates the three step modules.

    The fake reproduces the real step contract that matters for path handling:
    each step runs with its own ``cwd`` (the benchmark checkout) and resolves
    relative path arguments against *that* cwd. step3 reads ``--dedup-groups``
    relative to its cwd and — mirroring the real module — writes no
    ``evaluations.json`` when that path does not resolve to an existing file.
    """

    def fake_run(cmd, **kwargs):
        cwd = kwargs["cwd"]
        # The child's real working directory == parent cwd joined with the
        # ``cwd`` argument (an absolute ``cwd`` wins, exactly like a process).
        child_wd = Path.cwd() / cwd
        out_dir = child_wd / "results" / model_dir_name
        module = cmd[cmd.index("-m") + 1]
        if module.endswith("step2_extract_comments"):
            (out_dir / "candidates.json").write_text("{}")
        elif module.endswith("step2_5_dedup_candidates"):
            (out_dir / "dedup_groups.json").write_text("{}")
        elif module.endswith("step3_judge_comments"):
            dedup_arg = cmd[cmd.index("--dedup-groups") + 1]
            # step3 resolves --dedup-groups against ITS OWN cwd (child_wd).
            if (child_wd / dedup_arg).exists():
                (out_dir / "evaluations.json").write_text(
                    '{"https://x/pull/1": {"daydream": {"tp": 1, "fp": 0, "fn": 0}}}'
                )
            # else: mimic the real module's early `return` — write nothing.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return fake_run


def test_run_scoring_with_relative_benchmark_repo_resolves_dedup_path(tmp_path, monkeypatch):
    """Real-path regression: a RELATIVE benchmark_repo must still let step3 find
    its --dedup-groups file.

    Each step runs with ``cwd=benchmark_repo``; a benchmark-repo-relative
    ``--dedup-groups`` value (e.g. ``../code-review-benchmark/offline/results/…``)
    is re-interpreted against that cwd, doubles up, and misses — so step3 exits 0
    without writing ``evaluations.json`` and run_scoring raises
    ``BenchmarkArtifactError``. The fix resolves benchmark_repo to an absolute
    path so the dedup argument is cwd-independent. Drives the real run_scoring
    path construction, step ordering, artifact check, and parse; only the
    external judge subprocess is faked.
    """
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    model = "anthropic/claude-opus-4.5"
    model_dir_name = "anthropic_claude-opus-4.5"

    (tmp_path / "bench" / "results" / model_dir_name).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("daydream.benchmark.score.subprocess.run", _emulating_fake_run(model_dir_name))

    # benchmark_repo passed RELATIVE while the process cwd is its parent — the
    # exact shape the harness hits (`--benchmark-repo ../code-review-benchmark/offline`).
    scores = run_scoring(Path("bench"), model, pr_count=2)

    assert scores.scored_pr_count == 1
    assert scores.total_tp == 1
    # The dedup file step3 was pointed at actually exists on disk.
    assert (tmp_path / "bench" / "results" / model_dir_name / "evaluations.json").exists()


def test_run_scoring_relative_repo_regression_fails_without_absolute_resolution(tmp_path, monkeypatch):
    """Guard the fix directly: if benchmark_repo were left relative, step3's
    dedup path would miss and run_scoring would raise. We assert the post-fix
    behavior (no raise) here; the companion test above asserts the parsed
    scores. Together they pin the absolute-path resolution in place."""
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    model = "anthropic/claude-opus-4.5"
    model_dir_name = "anthropic_claude-opus-4.5"
    (tmp_path / "bench" / "results" / model_dir_name).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("daydream.benchmark.score.subprocess.run", _emulating_fake_run(model_dir_name))
    try:
        run_scoring(Path("bench"), model, pr_count=2)
    except BenchmarkArtifactError as exc:  # pragma: no cover - only on regression
        pytest.fail(f"relative benchmark_repo broke step3 dedup resolution: {exc}")

from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.benchmark.score import (
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    BenchmarkArtifactError,
    JudgeEnvError,
    JudgeFailedError,
    model_results_dir,
    parse_daydream_scores,
    preflight_judge_env,
    resolve_judge_model,
    run_scoring,
)

URL = "https://x/pull/1"


@pytest.mark.parametrize(
    ("api_key", "pinned_base_url", "expected_base_url"),
    [
        # OpenRouter key, no pin → routed to OpenRouter (the 401 regression).
        ("sk-or-v1-realkey", None, "https://openrouter.ai/api/v1"),
        # OpenRouter key, explicit pin → preserved (auto-route only fills UNSET).
        ("sk-or-v1-realkey", "https://api.withmartian.com/v1", "https://api.withmartian.com/v1"),
        # Non-OpenRouter key → left to the upstream default, never forced to OpenRouter.
        ("sk-martian-realkey", None, None),
    ],
)
def test_run_scoring_judge_base_url_routing(tmp_path, monkeypatch, api_key, pinned_base_url, expected_base_url):
    """The judge base URL forwarded to each step: an OpenRouter (sk-or-) key with no
    pinned base URL is routed to OpenRouter's host (else the upstream Martian default
    401s it); an explicit pin always wins; a non-OpenRouter key is left untouched."""
    monkeypatch.setenv("MARTIAN_API_KEY", api_key)
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    if pinned_base_url is None:
        monkeypatch.delenv(JUDGE_BASE_URL_ENV, raising=False)
    else:
        monkeypatch.setenv(JUDGE_BASE_URL_ENV, pinned_base_url)

    envs: list[dict] = []
    rdir = tmp_path / "results" / "anthropic_claude-opus-4-5-20251101"
    rdir.mkdir(parents=True)
    (rdir / "evaluations.json").write_text('{"%s": {"daydream-glm": {"tp":1,"fp":0,"fn":0}}}' % URL)

    def fake_run(cmd, **k):
        envs.append(k["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.benchmark.score.subprocess.run", fake_run)
    run_scoring(tmp_path, "anthropic/claude-opus-4-5-20251101", tool="daydream-glm")

    assert envs
    if expected_base_url is None:
        assert all(JUDGE_BASE_URL_ENV not in e for e in envs)
    else:
        assert all(e[JUDGE_BASE_URL_ENV] == expected_base_url for e in envs)


def test_resolve_judge_model(monkeypatch):
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    with pytest.raises(JudgeEnvError, match="Judge model unspecified"):
        resolve_judge_model(None)  # no hardcoded default
    monkeypatch.setenv(JUDGE_MODEL_ENV, "anthropic/claude-opus-4-5-20251101")
    assert resolve_judge_model(None) == "anthropic/claude-opus-4-5-20251101"


def test_run_scoring_passes_custom_tool_and_judge_env_to_each_step(tmp_path, monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    calls = []
    envs = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        envs.append(k["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.benchmark.score.subprocess.run", fake_run)
    rdir = tmp_path / "results" / "anthropic_claude-opus-4-5-20251101"
    rdir.mkdir(parents=True)
    (rdir / "evaluations.json").write_text('{"%s": {"daydream-glm": {"tp":1,"fp":0,"fn":0}}}' % URL)
    scores = run_scoring(tmp_path, "anthropic/claude-opus-4-5-20251101", tool="daydream-glm")
    assert all(c[c.index("--tool")+1] == "daydream-glm" for c in calls)
    # Every step is told which judge model to run via MARTIAN_MODEL — the same
    # value that names the results dir we read from.
    assert all(e[JUDGE_MODEL_ENV] == "anthropic/claude-opus-4-5-20251101" for e in envs)
    assert scores.scored_pr_count == 1                  # parse keyed off the custom label


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


#: The verbatim judge error from the daydream-glm incident: MARTIAN_MODEL was the
#: Anthropic-style dated id, which the OpenRouter gateway rejects on every call.
_MODEL_ID_ERROR = "anthropic/claude-opus-4-5-20251101 is not a valid model ID"

#: A leaf shaped exactly like the daydream-glm incident: the judge errored on
#: every one of the 17 candidates × 5 golden = 85 comparisons, so tp/fp/fn
#: collapse to a clean-looking zero that is really a total judge failure. step3
#: stores the real per-comparison error text in the ``errors`` list.
_ALL_ERRORED_LEAF = {
    "tp": 0, "fp": 17, "fn": 5,
    "errors_count": 85, "total_candidates": 17, "total_golden": 5,
    "precision": 0.0, "recall": 0.0,
    "errors": [{"golden": "g", "candidate": "c", "error": _MODEL_ID_ERROR} for _ in range(85)],
}


def test_parse_daydream_scores_raises_with_verbatim_judge_error():
    """A wholesale judge failure must surface the REAL recorded error, not a guess.

    Reproduces the daydream-glm incident: every judge call errored, so the harness
    reported precision=recall=0.000 — indistinguishable from a genuinely poor
    review. The guard flips that into a hard `JudgeFailedError` whose message
    quotes the actual step3 error string and never invents a cause.
    """
    evals = {URL: {"daydream-glm": dict(_ALL_ERRORED_LEAF)}}
    with pytest.raises(JudgeFailedError) as e:
        parse_daydream_scores(evals, tool="daydream-glm")
    msg = str(e.value)
    assert "85/85" in msg
    assert _MODEL_ID_ERROR in msg and "(85×)" in msg  # verbatim error + count
    assert "Most likely" not in msg and "401" not in msg  # no guessed cause


def test_parse_daydream_scores_failure_without_error_detail_does_not_invent_cause():
    """An older corpus leaf may carry errors_count but no ``errors`` text. The guard
    must still fire, report the count, and NOT fabricate a 401/credential cause."""
    leaf = {k: v for k, v in _ALL_ERRORED_LEAF.items() if k != "errors"}
    evals = {URL: {"daydream-glm": leaf}}
    with pytest.raises(JudgeFailedError) as e:
        parse_daydream_scores(evals, tool="daydream-glm")
    msg = str(e.value)
    assert "85/85" in msg
    assert "no per-comparison error text" in msg
    assert "401" not in msg and "Most likely" not in msg


def test_parse_daydream_scores_genuine_zero_with_no_errors_does_not_raise():
    """A real zero (judge ran cleanly, nothing matched) must NOT trip the guard.

    The daydream review produced candidates and the judge compared all of them
    without error — they simply did not match the golden set. errors_count==0, so
    the scores are trustworthy and the aggregate zero stands.
    """
    evals = {URL: {"daydream": {
        "tp": 0, "fp": 17, "fn": 5,
        "errors_count": 0, "total_candidates": 17, "total_golden": 5,
        "precision": 0.0, "recall": 0.0,
    }}}
    s = parse_daydream_scores(evals)
    assert s.scored_pr_count == 1
    assert s.precision == 0.0 and s.recall == 0.0
    assert s.total_errors == 0 and s.total_comparisons == 85


def test_parse_daydream_scores_no_candidates_does_not_raise():
    """A PR where daydream emitted nothing (0 candidates) is a legit zero, not a
    judge failure: no comparisons are attempted, so the ratio guard cannot fire."""
    evals = {URL: {"daydream": {
        "tp": 0, "fp": 0, "fn": 5,
        "errors_count": 0, "total_candidates": 0, "total_golden": 5,
    }}}
    s = parse_daydream_scores(evals)
    assert s.total_comparisons == 0
    assert s.recall == 0.0


def test_parse_daydream_scores_below_threshold_errors_do_not_raise():
    """A few transient judge errors (below the ratio threshold) are tolerated;
    the surviving comparisons still produce a usable score."""
    evals = {URL: {"daydream": {
        "tp": 2, "fp": 1, "fn": 1,
        "errors_count": 3, "total_candidates": 3, "total_golden": 3,  # 3/9 ≈ 0.33 < 0.5
    }}}
    s = parse_daydream_scores(evals)
    assert s.total_errors == 3 and s.total_comparisons == 9
    assert s.total_tp == 2


def test_run_scoring_raises_judge_failed_when_evaluations_all_errored(tmp_path, monkeypatch):
    """Real-path: drive run_scoring end-to-end against an on-disk evaluations.json
    where every comparison errored, and assert it raises `JudgeFailedError`.

    Only the external judge subprocesses are faked (they 401 in reality); the
    production path — step ordering, results-dir resolution, artifact read, and
    parse — runs for real and must convert the silent zero into a loud failure.
    """
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-bad")
    model = "anthropic/claude-opus-4-5-20251101"
    rdir = tmp_path / "results" / "anthropic_claude-opus-4-5-20251101"
    rdir.mkdir(parents=True)
    import json as _json

    def fake_run(cmd, **k):
        (rdir / "evaluations.json").write_text(_json.dumps({URL: {"daydream-glm": dict(_ALL_ERRORED_LEAF)}}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.benchmark.score.subprocess.run", fake_run)
    with pytest.raises(JudgeFailedError):
        run_scoring(tmp_path, model, pr_count=1, tool="daydream-glm")


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

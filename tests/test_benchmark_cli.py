"""Tests for the ``daydream bench`` CLI subcommand.

Covers an arg-parse unit test for ``_bench_config_from_argv`` and tier-3
real-path tests through the installed ``daydream`` console script.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from daydream.benchmark.cli import (
    _bench_config_from_argv,
    _build_benchmark_parser,
    _format_elapsed,
    _load_bench_dotenv,
)


def test_benchmark_parser_has_build_harbor_and_compiled():
    parser = _build_benchmark_parser()
    build_args = parser.parse_args(["build-harbor", "/workspace", "--daydream-wheel", "/d.whl"])
    assert build_args.subcommand == "build-harbor"
    assert build_args.daydream_wheel == Path("/d.whl")
    validate_args = parser.parse_args(["validate", "/workspace", "--compiled"])
    assert validate_args.compiled is True


def test_calibrate_judge_help_describes_diagnostic_agreement():
    parser = _build_benchmark_parser()
    help_text = parser._subparsers._group_actions[0].choices["calibrate-judge"].format_help()
    assert "diagnostic" in help_text.lower() or "agreement" in help_text.lower()
    assert "calibrat" in help_text.lower()          # still called calibrate-judge
    assert "unverified" in help_text.lower()


def test_docs_distinguish_diagnostics_from_oracle_gate():
    changelog = Path(__file__).parents[1] / "CHANGELOG.md"
    bench_doc = Path(__file__).parents[1] / "docs" / "benchmark.md"
    text = changelog.read_text() + bench_doc.read_text()
    assert "Oracle" in text and "diagnostic" in text.lower()
    assert "self-match" in text.lower() or "reward" in text.lower()
    assert "OpenRouter" in text and "data-handling" in text.lower()


def test_benchmark_build_harbor_real_cli_entry(tmp_path, fake_gh, capsys):
    import importlib.metadata

    pytest.importorskip("harbor")
    from daydream.benchmark.cli import _handle_benchmark_command
    from tests.test_benchmark_harbor_build import _seed_ready_workspace

    ws, _, _ = _seed_ready_workspace(tmp_path, fake_gh)
    version = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{version}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    code = _handle_benchmark_command([
        "build-harbor", str(ws), "--daydream-wheel", str(wheel)
    ])
    assert code == 0
    assert (ws / "harbor/benchmark.lock.json").is_file()
    assert "built Harbor dataset" in capsys.readouterr().out


def test_format_elapsed():
    assert _format_elapsed(45.4) == "45s"
    assert _format_elapsed(252) == "4m12s"


def test_load_bench_dotenv_populates_environ(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MARTIAN_API_KEY=sk-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARTIAN_API_KEY", raising=False)
    _load_bench_dotenv()
    assert os.environ["MARTIAN_API_KEY"] == "sk-from-dotenv"


def test_bench_parser_defaults_and_flags(tmp_path, monkeypatch):
    # Hermetic: chdir to a config-free dir so the built-in defaults are exercised,
    # not whatever [tool.daydream.bench] the repo's own pyproject happens to carry.
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--only", "grafana", "--no-score"])
    assert cfg.benchmark_repo == Path("/b") and cfg.only == "grafana" and cfg.score is False
    assert cfg.model is None  # no hardcoded default; judge model comes from --model or route-specific env


def test_bench_parser_accepts_direct_anthropic_judge_route(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv([
        "--benchmark-repo", "/b",
        "--judge-route", "anthropic-direct",
        "--model", "claude-opus-4-5-20251101",
        "--no-score",
    ])
    assert cfg.judge_route == "anthropic-direct"
    assert cfg.model == "claude-opus-4-5-20251101"


def test_bench_parser_accepts_openai_compatible_judge_route(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv([
        "--benchmark-repo", "/b",
        "--judge-route", "openai-compatible",
        "--model", "gpt-5.6-luna",
        "--no-score",
    ])
    assert cfg.judge_route == "openai-compatible"
    assert cfg.model == "gpt-5.6-luna"


def test_bench_config_has_reviewer_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--no-score"])
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_model is None
    assert cfg.reviewer_provider is None
    assert cfg.tool_label == "daydream"


def test_reviewer_flags_reach_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv([
        "--benchmark-repo", "/b", "--no-score",
        "--reviewer-backend", "pi", "--reviewer-model", "glm-5.2",
        "--reviewer-provider", "openrouter", "--tool-label", "daydream-glm",
    ])
    assert (cfg.reviewer_backend, cfg.reviewer_model, cfg.reviewer_provider, cfg.tool_label) \
        == ("pi", "glm-5.2", "openrouter", "daydream-glm")


# --reviewer-backend must accept every backend the main CLI accepts (incl. osprey).
REVIEWER_BACKENDS = ["claude", "codex", "pi", "osprey"]


@pytest.mark.parametrize("backend", REVIEWER_BACKENDS, ids=lambda name: name)
def test_reviewer_backend_accepts_all_cli_backends(tmp_path, monkeypatch, backend):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv([
        "--benchmark-repo", "/b", "--no-score",
        "--reviewer-backend", backend, "--tool-label", "daydream-x",
    ])
    assert cfg.reviewer_backend == backend


def test_config_supplies_benchmark_repo_when_flag_omitted(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.bench]\nbenchmark-repo = "/from/config"\n')
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--no-score"])  # no --benchmark-repo
    assert cfg.benchmark_repo == Path("/from/config")


def test_config_supplies_judge_route_when_flag_omitted(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.bench]\nbenchmark-repo="/b"\njudge-route="anthropic-direct"\n'
    )
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--no-score"])
    assert cfg.judge_route == "anthropic-direct"


def test_missing_benchmark_repo_everywhere_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config, no flag
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--no-score"])


def test_harvest_dir_flag_reaches_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--harvest-dir", "/h", "--no-score"])
    assert cfg.harvest_dir == Path("/h")
    assert cfg.benchmark_repo is None
    assert cfg.corpus_root == Path("/h")
    # Derived path defaults hang off the harvest dir, not a benchmark repo.
    assert cfg.cache_dir == Path("/h/.daydream-bench/cache")
    assert cfg.trajectory_dir == Path("/h/.daydream-bench/trajectories")


def test_harvest_dir_and_benchmark_repo_are_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--benchmark-repo", "/b", "--harvest-dir", "/h", "--no-score"])


def test_harvest_dir_with_martian_route_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The martian route shells the withmartian step modules, which only exist
    # inside that checkout; a harvested corpus must score in-process.
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--harvest-dir", "/h", "--score", "--judge-route", "martian"])
    cfg = _bench_config_from_argv(["--harvest-dir", "/h", "--score", "--judge-route", "anthropic-direct"])
    assert cfg.judge_route == "anthropic-direct"
    # The in-process OpenAI-compatible route parses on a harvested corpus too.
    cfg = _bench_config_from_argv(["--harvest-dir", "/h", "--score", "--judge-route", "openai-compatible"])
    assert cfg.judge_route == "openai-compatible"


def test_harvest_dir_config_file_fallback(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.bench]\nharvest-dir = "/from/config"\n')
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--no-score"])  # no --harvest-dir flag
    assert cfg.harvest_dir == Path("/from/config") and cfg.benchmark_repo is None


def test_reviewer_preset_resolves_and_derives_label(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.bench]\nbenchmark-repo = "/b"\n'
        '[tool.daydream.bench.reviewers.glm]\nbackend="pi"\nmodel="z-ai/glm-5.2"\nprovider="openrouter"\n'
    )
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--reviewer", "glm", "--no-score"])
    assert (cfg.reviewer_backend, cfg.reviewer_model, cfg.reviewer_provider, cfg.tool_label) \
        == ("pi", "z-ai/glm-5.2", "openrouter", "daydream-glm")
    cfg2 = _bench_config_from_argv(["--reviewer", "glm", "--reviewer-model", "x", "--no-score"])
    assert cfg2.reviewer_model == "x"  # explicit flag overrides preset


@pytest.mark.parametrize(
    ("config_text", "reviewer"),
    [
        ('[tool.daydream.bench]\nbenchmark-repo="/b"\n', "nope"),
        ('[tool.daydream.bench]\nbenchmark-repo="/b"\n[tool.daydream.bench.reviewers]\nglm="not-a-table"\n', "glm"),
        ('[tool.daydream.bench]\nbenchmark-repo="/b"\nreviewers="oops"\n', "glm"),
    ],
    ids=["unknown-preset", "malformed-preset", "non-table-reviewers"],
)
def test_reviewer_preset_errors(tmp_path, monkeypatch, config_text, reviewer):
    """Terminate configuration parsing for missing or malformed reviewer presets."""
    (tmp_path / "pyproject.toml").write_text(config_text)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--reviewer", reviewer, "--no-score"])


def test_malformed_reviewer_preset_fails_through_compiled_entrypoint(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.bench]\nbenchmark-repo="/b"\n[tool.daydream.bench.reviewers]\nglm="not-a-table"\n'
    )
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--reviewer", "glm", "--no-score"],  # noqa: S607 - daydream is a trusted command
        capture_output=True,
        text=True,
        cwd=tmp_path,  # the malformed pyproject.toml lives here; bench reads config from cwd
    )
    assert r.returncode != 0 and "unknown --reviewer 'glm'" in (r.stdout + r.stderr)


@pytest.mark.parametrize(
    "override",
    [
        ["--reviewer-backend", "pi"],
        ["--reviewer-model", "glm-5.2"],
        ["--reviewer-provider", "openrouter"],
    ],
)
def test_reviewer_override_without_label_errors(override):
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--benchmark-repo", "/b", "--no-score", *override])


def test_reviewer_override_with_explicit_label_ok():
    cfg = _bench_config_from_argv(
        ["--benchmark-repo", "/b", "--no-score", "--reviewer-backend", "pi", "--tool-label", "daydream-glm"]
    )
    assert cfg.tool_label == "daydream-glm"


def test_reviewer_override_without_label_fails_through_compiled_entrypoint(tmp_path):
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--benchmark-repo", str(tmp_path),  # noqa: S607 - daydream is a trusted command
         "--no-score", "--reviewer-backend", "pi"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode != 0 and "--tool-label" in (r.stdout + r.stderr)


def test_trials_flag_parsed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--no-score", "--trials", "3"])
    assert cfg.trials == 3


def test_trials_defaults_to_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--no-score"])
    assert cfg.trials == 1


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_trials_rejects_non_positive(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--benchmark-repo", "/b", "--no-score", "--trials", bad])


def test_trials_config_file_fallback(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.bench]\nbenchmark-repo="/b"\ntrials=5\n')
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--no-score"])  # no --trials flag
    assert cfg.trials == 5


def test_trials_flag_overrides_config_file(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.bench]\nbenchmark-repo="/b"\ntrials=5\n')
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--no-score", "--trials", "2"])
    assert cfg.trials == 2


@pytest.mark.parametrize(
    "toml_value",
    [
        pytest.param("2.5", id="float"),
        pytest.param('"3"', id="string"),
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
    ],
)
def test_trials_config_file_rejects_invalid_value(tmp_path, monkeypatch, toml_value):
    """Reject non-positive or non-integer trial counts from project configuration."""
    (tmp_path / "pyproject.toml").write_text(f'[tool.daydream.bench]\nbenchmark-repo="/b"\ntrials={toml_value}\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--no-score"])


def test_bench_parser_accepts_positive_limit():
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--limit", "3"])
    assert cfg.limit == 3


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_bench_parser_rejects_non_positive_limit(bad):
    with pytest.raises(SystemExit):
        _bench_config_from_argv(["--benchmark-repo", "/b", "--limit", bad])


def test_bench_non_positive_limit_fails_through_compiled_entrypoint(tmp_path):
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--benchmark-repo", str(tmp_path), "--limit", "0"],  # noqa: S607 - daydream is a trusted command
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "--limit must be a positive integer" in (r.stdout + r.stderr)


@pytest.mark.parametrize(
    ("route_args", "env_overrides", "credential"),
    [
        ([], {}, "MARTIAN_API_KEY"),
        (["--judge-route", "anthropic-direct"], {"MARTIAN_MODEL": "claude-opus-4-5-20251101"}, "ANTHROPIC_API_KEY"),
    ],
    ids=["martian", "anthropic-direct"],
)
def test_compiled_entrypoint_preflights_credentials(tmp_path, route_args, env_overrides, credential):
    """Fail before scoring when the selected judge route lacks credentials."""
    env = {**os.environ, **env_overrides}
    env.pop(credential, None)
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--benchmark-repo", str(tmp_path), *route_args, "--score"],  # noqa: S607
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert r.returncode != 0 and credential in (r.stdout + r.stderr)


def test_benchmark_docs_name_direct_anthropic_judge_route():
    text = Path("docs/benchmark.md").read_text()
    assert "--judge-route anthropic-direct" in text
    assert "ANTHROPIC_API_KEY" in text
    assert "`MARTIAN_BASE_URL` is invalid" in text
    assert "--reviewer-backend" in text and "--model" in text
    # The in-process OpenAI-compatible route is documented too.
    assert "--judge-route openai-compatible" in text
    assert "OPENAI_API_KEY" in text
    assert "OPENAI_BASE_URL" in text


def test_bench_dotenv_autoloads_credential_through_compiled_entrypoint(tmp_path):
    (tmp_path / ".env").write_text("MARTIAN_API_KEY=sk-from-dotenv\n")
    env = {**os.environ}
    env.pop("MARTIAN_API_KEY", None)
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--benchmark-repo", str(tmp_path), "--score"],  # noqa: S607 - daydream is a trusted command
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # the .env lives here; the real bench entry must auto-load it from cwd
    )
    assert "MARTIAN_API_KEY is not set" not in (r.stdout + r.stderr)


def _objective_ws(tmp_path):
    """A complete, consistent run seeded via the Task-1/2 test helper."""
    from tests.test_benchmark_objective import _complete_ws
    return _complete_ws(tmp_path)


def test_objective_cli_writes_json_atomically_and_summary(tmp_path, capsys):
    from daydream.benchmark.cli import _handle_benchmark_command

    ws = _objective_ws(tmp_path)
    out = tmp_path / "obj.json"
    code = _handle_benchmark_command([
        "objective", str(ws), "--run-id", "run-1", "--json", str(out),
    ])
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["run_id"] == "run-1" and "f1" in doc["objective"]
    captured = capsys.readouterr().out
    assert "run-1" in captured   # concise local summary by default


def test_objective_cli_json_dash_keeps_stdout_pure_json(tmp_path, capsys):
    from daydream.benchmark.cli import _handle_benchmark_command

    ws = _objective_ws(tmp_path)
    code = _handle_benchmark_command([
        "objective", str(ws), "--run-id", "run-1", "--json", "-",
    ])
    assert code == 0
    captured = capsys.readouterr()
    # Issue #888 machine-readable '-' mode: stdout must be exactly the JSON blob
    # (no interleaved human summary); the summary routes to stderr.
    doc = json.loads(captured.out)
    assert doc["run_id"] == "run-1" and "f1" in doc["objective"]
    assert "objective run-1:" in captured.err   # human summary went to stderr


def test_objective_cli_failure_leaves_output_unchanged(tmp_path):
    from daydream.benchmark.cli import _handle_benchmark_command
    from daydream.benchmark.harbor import run as run_mod

    ws = _objective_ws(tmp_path)
    run_mod.ledger_mark(ws, "run-1", state="running")   # non-terminal -> fail closed
    out = tmp_path / "obj.json"
    out.write_text("SENTINEL")
    code = _handle_benchmark_command([
        "objective", str(ws), "--run-id", "run-1", "--json", str(out),
    ])
    assert code == 1
    assert out.read_text() == "SENTINEL"


def test_aggregate_cli_writes_json_and_prints_digest(tmp_path, capsys):
    from daydream.benchmark.cli import _handle_benchmark_command
    from tests.test_benchmark_objective import _complete_ws_at, _reward
    a = _complete_ws_at(tmp_path, "a", "r1", [_reward(tp=1, fp=0, fn=0)])
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=0, fn=0)])
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"},
        {"workspace": str(b), "run_id": "r2"}]}))
    out = tmp_path / "agg.json"
    code = _handle_benchmark_command(["aggregate", str(manifest), "--json", str(out)])
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["experiment_id"]
    assert doc["profile_digest"]
    assert "micro_precision" in doc["objective"]
    assert "d"*64 in capsys.readouterr().out   # digest always printed


def test_aggregate_cli_fails_closed_leaves_output_unchanged(tmp_path):
    from daydream.benchmark.cli import _handle_benchmark_command
    from tests.test_benchmark_objective import _complete_ws_at, _reward
    a = _complete_ws_at(tmp_path, "a", "r1", [_reward(tp=1, fp=0, fn=0)], digest="d"*64)
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=0, fn=0)], digest="e"*64)
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"},
        {"workspace": str(b), "run_id": "r2"}]}))
    out = tmp_path / "agg.json"
    out.write_text("SENTINEL")
    code = _handle_benchmark_command(["aggregate", str(manifest), "--json", str(out)])
    assert code == 1
    assert out.read_text() == "SENTINEL"


def test_bench_help_lists_flags():
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--help"], capture_output=True, text=True  # noqa: S607 - daydream is a trusted command
    )
    assert r.returncode == 0 and "--benchmark-repo" in r.stdout

"""Tests for the ``daydream bench`` CLI subcommand.

Covers an arg-parse unit test for ``_bench_config_from_argv`` and tier-3
real-path tests through the installed ``daydream`` console script.
"""

import os
import subprocess
from pathlib import Path

import pytest

from daydream.benchmark.cli import _bench_config_from_argv, _format_elapsed, _load_bench_dotenv


def test_format_elapsed():
    assert _format_elapsed(45.4) == "45s"
    assert _format_elapsed(252) == "4m12s"


def test_load_bench_dotenv_populates_environ(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MARTIAN_API_KEY=sk-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MARTIAN_API_KEY", raising=False)
    _load_bench_dotenv()
    assert os.environ["MARTIAN_API_KEY"] == "sk-from-dotenv"


def test_bench_parser_defaults_and_flags():
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--only", "grafana", "--no-score"])
    assert cfg.benchmark_repo == Path("/b") and cfg.only == "grafana" and cfg.score is False
    assert cfg.model is None  # no hardcoded default; judge model comes from --model or MARTIAN_MODEL


def test_bench_config_has_reviewer_defaults():
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--no-score"])
    assert cfg.reviewer_backend is None
    assert cfg.reviewer_model is None
    assert cfg.reviewer_provider is None
    assert cfg.tool_label == "daydream"


def test_reviewer_flags_reach_config():
    cfg = _bench_config_from_argv([
        "--benchmark-repo", "/b", "--no-score",
        "--reviewer-backend", "pi", "--reviewer-model", "glm-5.2",
        "--reviewer-provider", "openrouter", "--tool-label", "daydream-glm",
    ])
    assert (cfg.reviewer_backend, cfg.reviewer_model, cfg.reviewer_provider, cfg.tool_label) \
        == ("pi", "glm-5.2", "openrouter", "daydream-glm")


def test_config_supplies_benchmark_repo_when_flag_omitted(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.bench]\nbenchmark-repo = "/from/config"\n')
    monkeypatch.chdir(tmp_path)
    cfg = _bench_config_from_argv(["--no-score"])  # no --benchmark-repo
    assert cfg.benchmark_repo == Path("/from/config")


def test_missing_benchmark_repo_everywhere_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config, no flag
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


def test_bench_subcommand_preflights_through_compiled_entrypoint(tmp_path):
    env = {**os.environ}
    env.pop("MARTIAN_API_KEY", None)
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--benchmark-repo", str(tmp_path), "--score"],  # noqa: S607 - daydream is a trusted command
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # isolate from any developer .env auto-loaded at bench entry
    )
    assert r.returncode != 0 and "MARTIAN_API_KEY" in (r.stdout + r.stderr)


def test_bench_help_lists_flags():
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--help"], capture_output=True, text=True  # noqa: S607 - daydream is a trusted command
    )
    assert r.returncode == 0 and "--benchmark-repo" in r.stdout

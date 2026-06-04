"""Tests for the ``daydream bench`` CLI subcommand.

Covers an arg-parse unit test for ``_bench_config_from_argv`` and tier-3
real-path tests through the installed ``daydream`` console script.
"""

import os
import subprocess
from pathlib import Path

from daydream.cli import _bench_config_from_argv


def test_bench_parser_defaults_and_flags():
    cfg = _bench_config_from_argv(["--benchmark-repo", "/b", "--only", "grafana", "--no-score"])
    assert cfg.benchmark_repo == Path("/b") and cfg.only == "grafana" and cfg.score is False
    assert cfg.model == "anthropic/claude-opus-4.5"


def test_bench_subcommand_preflights_through_compiled_entrypoint(tmp_path):
    env = {**os.environ}
    env.pop("MARTIAN_API_KEY", None)
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--benchmark-repo", str(tmp_path), "--score"],  # noqa: S607 - daydream is a trusted command
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode != 0 and "MARTIAN_API_KEY" in (r.stdout + r.stderr)


def test_bench_help_lists_flags():
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "bench", "--help"], capture_output=True, text=True  # noqa: S607 - daydream is a trusted command
    )
    assert r.returncode == 0 and "--benchmark-repo" in r.stdout

"""Hermetic suite for the `daydream benchmark run` supervisor (issue #781).

Task 1 (Step 1) failing tests: parser + dispatch routing.
"""
from pathlib import Path

from daydream.benchmark.cli import _build_benchmark_parser, _handle_benchmark_command


def test_benchmark_parser_has_run_subcommand():
    parser = _build_benchmark_parser()
    args = parser.parse_args(["run", "/ws", "--oracle", "--yes"])
    assert args.subcommand == "run"
    assert args.dir == Path("/ws")
    assert args.oracle is True and args.yes is True
    plain = parser.parse_args(["run", "/ws"])
    assert plain.oracle is False and plain.yes is False


def test_handle_benchmark_run_routes_to_supervisor(tmp_path, monkeypatch):
    import daydream.benchmark.harbor.run as run_mod

    captured = {}

    def fake_run_run(ws, *, oracle, yes, env):
        captured["ws"] = Path(ws)
        captured["oracle"] = oracle
        captured["yes"] = yes
        captured["env"] = env
        return 0

    monkeypatch.setattr(run_mod, "run_run", fake_run_run)
    code = _handle_benchmark_command(["run", str(tmp_path), "--yes"])
    assert code == 0
    assert captured["ws"] == tmp_path
    assert captured["yes"] is True and captured["oracle"] is False
    assert "DAYDREAM_REVIEW_MODEL" in captured["env"]  # env threaded through
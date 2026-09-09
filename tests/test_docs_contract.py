import contextlib
import io
from pathlib import Path

import pytest

import daydream.extensions as ext
from daydream.benchmark.cli import _build_benchmark_parser
from daydream.cli import _parse_args
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins

ROOT = Path(__file__).resolve().parents[1]


def test_readme_run_examples_parse() -> None:
    readme = (ROOT / "README.md").read_text()
    # the "common commands" code fence — the required run examples
    section = readme.split("Use the common commands for the common tasks:", 1)[1]
    fence = section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    lines = [
        line.strip()
        for line in fence.splitlines()
        if line.strip().startswith("daydream")
    ]
    assert lines, "the common-commands fence documents no daydream command"
    # each documented run example must be accepted by the production parser
    for line in lines:
        tokens = line.split()[1:]  # drop the leading 'daydream' verb
        if "#" in tokens:  # drop any inline comment
            tokens = tokens[: tokens.index("#")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            _parse_args(tokens)  # parses without SystemExit


def test_benchmark_objective_aggregate_parse() -> None:
    p = _build_benchmark_parser()                    # production parser
    p.parse_args(["objective", "./ws", "--run-id", "run-abc123", "--json", "-"])
    p.parse_args(["aggregate", "./suite.json", "--json", "-"])


def test_extensions_doc_claims_only_exposed() -> None:
    doc = (ROOT / "docs" / "extensions.md").read_text()
    reg = Registry()
    register_builtins(reg)
    # every public API symbol the doc's table names must exist in production
    table = doc.split("### Public API symbols")[1].split("### Discovery order")[0]
    for row in table.splitlines():
        if row.strip().startswith("| `"):
            symbol = row.split("`")[1]
            assert hasattr(ext, symbol), f"doc claims unexposed symbol {symbol!r}"
    # every prompt the doc's table names must be registered
    prompts = doc.split("### Prompts")[1].split("### Renderers")[0]
    for row in prompts.splitlines():
        if row.strip().startswith("| `"):
            name = row.split("`")[1]
            assert name in reg.prompt_names(), f"doc claims unexposed prompt {name!r}"


def test_help_exposes_native_surface(capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("--help", "--help-all"):
        with pytest.raises(SystemExit):
            _parse_args([flag])
        out = capsys.readouterr().out
        assert "--review-profile" in out and "--stack" in out
        assert "feedback" not in out and "--skill" not in out

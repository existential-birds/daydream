import contextlib
import io
from pathlib import Path

from daydream.cli import _parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_readme_profile_precedence_statement() -> None:
    readme = (ROOT / "README.md").read_text().lower()
    assert "daydream_review_profile" in readme
    assert "file_config.review_profile" in readme
    assert "built-in default" in readme  # terminal source, never 'packaged'
    assert "packaged" not in readme
    assert "host-owned" in readme  # host-owned invariants statement
    assert "outside the profile" in readme
    for host in ("backend", "provider", "model", "reasoning effort", "scoring"):
        assert host in readme


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
    assert any("--review-profile" in line for line in lines), (
        "missing explicit --review-profile example"
    )
    assert any("--stack" in line or line.startswith("daydream -s") for line in lines), (
        "missing explicit --stack example"
    )
    # each documented run example must be accepted by the production parser
    for line in lines:
        tokens = line.split()[1:]  # drop the leading 'daydream' verb
        if "#" in tokens:  # drop any inline comment
            tokens = tokens[: tokens.index("#")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            _parse_args(tokens)  # parses without SystemExit

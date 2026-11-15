"""Exercise ext validate via cli.main/sys.argv and an ext_dir package.

The real loader, API gate, and registry checks determine stdout and exit status.
"""
import re

import pytest

from tests.conftest import ExtDir
from tests.harness.console import collapse_panel_text
from tests.harness.scripts import cli_main as _run_main


def strip_ansi(text: str) -> str:
    """Normalize terminal styling for the CLI assertions below."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_ext_validate_rejects_previous_api(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str],
                                         monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('COLUMNS', "80")
    ext_dir.write_module("def register(r): ...\n", api_version=9)
    assert _run_main(["ext", "validate"]) == 1
    assert "must declare DAYDREAM_EXT_API = 10; found 9 " in collapse_panel_text(capsys)

def test_ext_validate_rejects_invalid_supervisor_registration(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str],
) -> None:
    ext_dir.write_module("def register(r): r.register_tool_supervisor(None)\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    assert "tool supervisor" in strip_ansi(capsys.readouterr().out).lower()

def test_ext_validate_broken_ref(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str]) -> None:
    ext_dir.write_module("def register(r):\n" "    r.set_flow('deep', ['ghost'])\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    assert "ghost" in strip_ansi(capsys.readouterr().out)

def test_ext_validate_reports_registered_renderer_count(
    ext_dir: ExtDir, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI summary reflects renderer slots added by a valid extension."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("HH_API_KEY", raising=False)
    ext_dir.write_module("def register(r): ...\n")
    assert _run_main(["ext", "validate"]) == 0
    baseline = strip_ansi(capsys.readouterr().out)
    assert "tool supervisor: none" in baseline.lower()
    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "def supervise(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def exporter(config):\n"
        "    raise AssertionError('must not instantiate during validation')\n"
        "def register(r):\n"
        "    r.register_tool_supervisor(supervise)\n"
        "    r.override_renderer('custom', lambda *args: 'custom')\n"
        "    r.register_trace_exporter('custom', exporter)\n"
    )
    assert _run_main(["ext", "validate"]) == 0
    out = strip_ansi(capsys.readouterr().out)
    def renderer_count(text: str) -> int:
        match = re.search(r"registry OK: .*?, (\d+) renderers", text)
        assert match is not None, text
        return int(match.group(1))

    assert "DAYDREAM_EXT_DIR" in out and "api version 10" in out.lower()
    assert "tool supervisor: registered" in out.lower()
    assert "trace exporters: langsmith, honeyhive, otlp, custom" in out
    assert "registry OK:" in out
    assert renderer_count(out) == renderer_count(baseline) + 1

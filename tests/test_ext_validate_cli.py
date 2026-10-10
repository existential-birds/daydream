"""Exercise ext validate via cli.main/sys.argv and an ext_dir package.

The real loader, API gate, and registry checks determine stdout and exit status.
"""
import re

import pytest

from tests.conftest import ExtDir
from tests.harness.console import collapse_panel_text
from tests.harness.scripts import cli_main as _run_main
from tests.test_integration import strip_ansi


def test_ext_validate_ok(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str]) -> None:
    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "def supervise(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r): r.register_tool_supervisor(supervise)\n"
    )
    rc = _run_main(["ext", "validate"])
    assert rc == 0
    out = strip_ansi(capsys.readouterr().out)
    assert "DAYDREAM_EXT_DIR" in out and "api version 8" in out.lower()
    assert "tool supervisor: registered" in out.lower()

def test_ext_validate_without_supervisor_reports_none(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str]) -> None:
    ext_dir.write_module("def register(r): ...\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 0
    out = strip_ansi(capsys.readouterr().out).lower()
    assert "tool supervisor: none" in out
    assert "supported: 8..8" in out


@pytest.mark.parametrize('api_version', [6, 7])
@pytest.mark.parametrize('width', [68, 80])
def test_ext_validate_rejects_previous_api(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str], api_version: int,
                                         width: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('COLUMNS', str(width))
    ext_dir.write_module("def register(r): ...\n", api_version=api_version)
    assert _run_main(["ext", "validate"]) == 1
    out = collapse_panel_text(capsys)
    assert f"DAYDREAM_EXT_API = {api_version}" in out
    assert "supports 8..8" in out

def test_ext_validate_rejects_invalid_supervisor_registration(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str],
) -> None:
    ext_dir.write_module("def register(r): r.register_tool_supervisor(None)\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    assert "tool supervisor" in strip_ansi(capsys.readouterr().out).lower()

def test_ext_validate_rejects_async_supervisor_registration(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str],
) -> None:
    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "async def supervise(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r): r.register_tool_supervisor(supervise)\n"
    )
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    out = strip_ansi(capsys.readouterr().out).lower()
    assert "tool supervisor" in out
    assert "synchronous" in out

def test_ext_validate_broken_ref(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str]) -> None:
    ext_dir.write_module("def register(r):\n" "    r.set_flow('deep', ['ghost'])\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    assert "ghost" in strip_ansi(capsys.readouterr().out)

def test_ext_validate_reports_registered_renderer_count(ext_dir: ExtDir, capsys: pytest.CaptureFixture[str],) -> None:
    """The CLI summary reflects renderer slots added by a valid extension."""
    ext_dir.write_module("def register(r): ...\n")
    assert _run_main(["ext", "validate"]) == 0
    baseline = strip_ansi(capsys.readouterr().out)
    ext_dir.write_module("def register(r):\n" "    r.override_renderer('custom', lambda *args: 'custom')\n")
    assert _run_main(["ext", "validate"]) == 0
    out = strip_ansi(capsys.readouterr().out)
    def renderer_count(text: str) -> int:
        match = re.search(r"registry OK: .*?, (\d+) renderers", text)
        assert match is not None, text
        return int(match.group(1))

    assert "registry OK:" in out
    assert renderer_count(out) == renderer_count(baseline) + 1

def test_bare_ext_prints_help_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run_main(["ext"])
    assert rc == 2
    assert "validate" in strip_ansi(capsys.readouterr().out)

def test_ext_validate_lists_exporters_without_initializing(
    ext_dir: ExtDir, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("HH_API_KEY", raising=False)
    ext_dir.write_module(
        "def exporter(config):\n"
        "    raise AssertionError('must not instantiate during validation')\n"
        "def register(r): r.register_trace_exporter('custom', exporter)\n"
    )
    assert _run_main(["ext", "validate"]) == 0
    out = strip_ansi(capsys.readouterr().out)
    assert "trace exporters: langsmith, honeyhive, otlp, custom" in out

"""Real-path tests for ``daydream ext validate``.

These drive ``cli.main`` through ``sys.argv`` (the production entrypoint,
matching ``tests/test_cli_corpus_namespace.py``) and assert the exit code
plus the user-visible stdout. The extension module comes from the ``ext_dir``
fixture (``$DAYDREAM_EXT_DIR`` seam), so the loader, version gate, and
registry resolve-check all run for real.
"""

import re
import sys

from daydream import cli

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for assertion comparisons."""
    return _ANSI_ESCAPE.sub("", text)


def _run_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


def test_ext_validate_ok(ext_dir, capsys) -> None:
    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "DAYDREAM_EXT_API = 2\n"
        "def supervise(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r): r.register_tool_supervisor(supervise)\n"
    )
    rc = _run_main(["ext", "validate"])
    assert rc == 0
    out = strip_ansi(capsys.readouterr().out)
    assert "DAYDREAM_EXT_DIR" in out and "api version 2" in out.lower()
    assert "tool supervisor: registered" in out.lower()


def test_ext_validate_without_supervisor_reports_none(ext_dir, capsys) -> None:
    ext_dir.write_module("DAYDREAM_EXT_API = 2\ndef register(r): ...\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 0
    assert "tool supervisor: none" in strip_ansi(capsys.readouterr().out).lower()


def test_ext_validate_rejects_invalid_supervisor_registration(ext_dir, capsys) -> None:
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 2\n"
        "def register(r): r.register_tool_supervisor(None)\n"
    )
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    assert "tool supervisor" in strip_ansi(capsys.readouterr().out).lower()


def test_ext_validate_rejects_async_supervisor_registration(ext_dir, capsys) -> None:
    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "DAYDREAM_EXT_API = 2\n"
        "async def supervise(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r): r.register_tool_supervisor(supervise)\n"
    )
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    out = strip_ansi(capsys.readouterr().out).lower()
    assert "tool supervisor" in out
    assert "synchronous" in out


def test_ext_validate_broken_ref(ext_dir, capsys) -> None:
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 2\n"
        "def register(r):\n"
        "    r.set_flow('deep', ['ghost'])\n"
    )
    rc = _run_main(["ext", "validate"])
    assert rc == 1
    assert "ghost" in strip_ansi(capsys.readouterr().out)


def test_ext_validate_reports_supported_range(ext_dir, capsys) -> None:
    ext_dir.write_module("DAYDREAM_EXT_API = 2\ndef register(r): ...\n")
    rc = _run_main(["ext", "validate"])
    assert rc == 0
    assert "supported: 1..2" in strip_ansi(capsys.readouterr().out).lower()


def test_bare_ext_prints_help_exits_2(capsys) -> None:
    rc = _run_main(["ext"])
    assert rc == 2
    assert "validate" in strip_ansi(capsys.readouterr().out)

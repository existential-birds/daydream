"""Command-line entry point of the scripted harness backends."""

from __future__ import annotations

import sys

from daydream import cli


def cli_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    raise AssertionError("cli.main() must exit via sys.exit")

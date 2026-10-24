"""Canonical Codex scripts with structured-output or recorded-byte replay.

Synthesized structured output replaces final agent-message text so the real
backend parser can extract it. Recorded raw_lines bypass all synthesis."""

from __future__ import annotations

import json
import sys
from typing import Any

from daydream import cli
from tests.contract._loaders import _build_codex_jsonl


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


def _with_structured_output(script: dict[str, Any]) -> dict[str, Any]:
    """Copy turns and replace final text with serialized structured output.
    Scripts without that payload retain their original identity."""
    structured = script.get("structured_output")
    if structured is None:
        return script
    turns = script["turns"]
    if not turns:
        raise AssertionError("structured_output requested but script has no turns to carry it")
    serialized = json.dumps(structured)
    new_turns = [dict(t) for t in turns]
    new_turns[-1]["text"] = serialized
    return {**script, "turns": new_turns}


def build_codex_jsonl_for_phase(script: dict[str, Any]) -> list[str]:
    """Synthesize canonical turns or replay recorded raw_lines verbatim.

    Raw replay cannot also request turns or structured_output synthesis."""
    raw_lines = script.get("raw_lines")
    if raw_lines is not None:
        if "turns" in script or "structured_output" in script:
            raise AssertionError("a raw_lines passthrough script must not also carry synthesized "
                "'turns' or 'structured_output'"
            )
        return list(raw_lines)
    return _build_codex_jsonl(_with_structured_output(script))
